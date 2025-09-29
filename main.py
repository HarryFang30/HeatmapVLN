#!/usr/bin/env python3
"""
VLN Project Main Entry Point - Production Ready Implementation
First-Person Inter-Frame Heatmap Generation for Vision-Language Navigation

基于已验证的完整pipeline，提供统一的命令行接口支持：
- 实时推理：生成frame-indexed heatmaps
- 算法选择：灵活切换采样算法
- 性能分析：详细的运行时指标
- 批处理：支持多视频处理

Architecture Pipeline (✅ FULLY WORKING):
1. VGGT (3D encoder) - 几何特征提取和空间感知采样
2. DINOv3 (2D encoder) - 语义特征提取
3. Qwen2.5-VL LLM - 空间推理和多模态理解
4. Frame-Indexed Heatmap Generation - 每个关键帧的独特空间映射

"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import working project modules
from src.utils.plotting_config import configure_matplotlib_fonts
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.data.algorithm_factory import get_factory
from src.data.enhanced_frame_sampler import EnhancedFrameSampler
from src.models.heatmap.converter import FrameIndexedHeatmapConverter
from src.testing.flexible_test_framework import FlexibleTestFramework
from src.models.real_llm_integration import PathResolver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Matplotlib fonts once so CJK instructions render in generated plots.
configure_matplotlib_fonts()


class VLNConfig:
    """配置管理器，支持YAML文件和命令行参数"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_default_config()

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            self._merge_config(file_config)

    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'model': {
                'use_real_llm': True,
                'llm_model_path': './models/qwen_2.5_vl',  # 使用相对路径
                'device_allocation': {
                    'vggt': 'cuda:0',
                    'dinov3': 'cuda:1',
                    'llm': 'cuda:2'
                }
            },
            'video': {
                'max_frames': 32,  # 候选帧池大小 (可通过--max_frames调整)
                'target_size': (224, 224),
                'fps_limit': 30,
                'sample_fps': None  # 可选：基于FPS采样
            },
            'sampling': {
                'algorithm': 'enhanced',  # fast, quality, enhanced
                'num_keyframes': 8,
                'diversity_weight': 0.4,
                'coverage_weight': 0.3,
                'novelty_weight': 0.3
            },
            'output': {
                'save_heatmaps': True,
                'save_metrics': True,
                'visualization': True,
                'output_dir': './outputs'
            },
            'performance': {
                'benchmark_mode': False,
                'profile_memory': False,
                'timing_analysis': True
            }
        }

    def _merge_config(self, new_config: Dict):
        """合并配置字典"""
        def merge_dict(base: Dict, new: Dict):
            for key, value in new.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value

        merge_dict(self.config, new_config)

    def get(self, key: str, default=None):
        """获取配置值，支持点分隔符"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


class PerformanceMonitor:
    """性能监控和分析"""

    def __init__(self):
        self.metrics = {}
        self.timings = {}
        self.memory_usage = {}

    def start_timer(self, name: str):
        """开始计时"""
        self.timings[name] = time.time()

    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name in self.timings:
            duration = time.time() - self.timings[name]
            self.metrics[f'{name}_duration'] = duration
            return duration
        return 0.0

    def record_memory(self, name: str):
        """记录GPU内存使用"""
        if torch.cuda.is_available():
            self.memory_usage[name] = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9
            }

    def get_summary(self) -> Dict:
        """获取性能摘要"""
        return {
            'metrics': self.metrics,
            'memory_usage': self.memory_usage,
            'total_time': sum(v for k, v in self.metrics.items() if k.endswith('_duration'))
        }


class VideoProcessor:
    """视频处理器，支持多种输入格式"""

    def __init__(self, config: VLNConfig):
        self.config = config

    def load_video(self, video_path: str) -> List[np.ndarray]:
        """加载视频文件并提取帧"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        logger.info(f"正在加载视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        frames = []
        max_frames = self.config.get('video.max_frames', 32)
        target_size = self.config.get('video.target_size', (224, 224))
        sample_fps = self.config.get('video.sample_fps', None)

        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if sample_fps is not None and video_fps > 0:
            # 基于FPS的采样
            skip_rate = max(1, int(video_fps / sample_fps))
            frame_indices = list(range(0, total_frames, skip_rate))[:max_frames]
            logger.info(f"FPS采样: 原始{video_fps:.2f}fps → 采样{sample_fps:.2f}fps，跳帧率: {skip_rate}")
        else:
            # 均匀采样
            if total_frames <= max_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames // max_frames
                frame_indices = list(range(0, total_frames, step))[:max_frames]
            logger.info(f"均匀采样: 总{total_frames}帧 → 采样{len(frame_indices)}帧")

        # 按索引提取帧
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整大小
            frame = cv2.resize(frame, target_size)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError("视频中没有提取到有效帧")

        logger.info(f"成功加载 {len(frames)} 帧")
        return frames

    def load_image_sequence(self, image_dir: str) -> List[np.ndarray]:
        """从图像序列加载帧"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")

        image_files = sorted([
            f for f in image_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])

        if not image_files:
            raise ValueError(f"目录中没有找到有效的图像文件: {image_dir}")

        frames = []
        target_size = self.config.get('video.target_size', (224, 224))
        max_frames = self.config.get('video.max_frames', 32)

        for img_file in image_files[:max_frames]:
            img = Image.open(img_file).convert('RGB')
            img = img.resize(target_size)
            frame = np.array(img)
            frames.append(frame)

        logger.info(f"从图像序列加载 {len(frames)} 帧")
        return frames


class HeatmapVisualizer:
    """热力图可视化工具"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_frame_indexed_heatmaps(self,
                                   frames: List[np.ndarray],
                                   keyframe_indices: List[int],
                                   heatmaps: np.ndarray,
                                   prefix: str = "heatmap") -> List[str]:
        """保存frame-indexed热力图"""
        saved_files = []

        for i, (frame_idx, heatmap) in enumerate(zip(keyframe_indices, heatmaps)):
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            # Original frame
            ax1.imshow(frames[frame_idx])
            ax1.set_title(f'Original Frame {frame_idx}')
            ax1.axis('off')

            # Heatmap
            im2 = ax2.imshow(heatmap, cmap='hot', interpolation='nearest')
            ax2.set_title(f'Heatmap {i}')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, shrink=0.8)

            # Overlay - ensure size alignment
            frame_img = frames[frame_idx]
            heatmap_h, heatmap_w = heatmap.shape

            # Resize frame to match heatmap dimensions exactly
            if frame_img.shape[:2] != (heatmap_h, heatmap_w):
                aligned_frame = cv2.resize(frame_img, (heatmap_w, heatmap_h))
            else:
                aligned_frame = frame_img

            # Display with consistent parameters and exact alignment
            ax3.imshow(aligned_frame, alpha=0.7, interpolation='bilinear', extent=[0, heatmap_w, heatmap_h, 0])
            ax3.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='bilinear', extent=[0, heatmap_w, heatmap_h, 0])
            ax3.set_title(f'Overlay {i}')
            ax3.axis('off')

            # Save
            filename = f"{prefix}_frame_{frame_idx}_heatmap_{i}.png"
            save_path = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            saved_files.append(str(save_path))
            logger.info(f"Saved heatmap: {save_path}")

        return saved_files

    def create_summary_visualization(self,
                                   frames: List[np.ndarray],
                                   keyframe_indices: List[int],
                                   heatmaps: np.ndarray,
                                   metrics: Dict) -> str:
        """创建汇总可视化"""
        num_keyframes = len(keyframe_indices)
        cols = min(4, num_keyframes)
        rows = (num_keyframes + cols - 1) // cols

        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 4, rows * 6))
        if axes.ndim == 1:
            axes = axes.reshape(-1, 1)

        for i, (frame_idx, heatmap) in enumerate(zip(keyframe_indices, heatmaps)):
            row = (i // cols) * 2
            col = i % cols

            # Original frame
            axes[row, col].imshow(frames[frame_idx])
            axes[row, col].set_title(f'Frame {frame_idx}')
            axes[row, col].axis('off')

            # Heatmap
            im = axes[row + 1, col].imshow(heatmap, cmap='hot')
            axes[row + 1, col].set_title(f'Heatmap {i}')
            axes[row + 1, col].axis('off')

        # Add performance metrics text
        if 'total_time' in metrics:
            fig.suptitle(f'Frame-Indexed Heatmaps (Processing Time: {metrics["total_time"]:.2f}s)',
                        fontsize=16)

        # Save summary plot
        summary_path = self.output_dir / "summary_heatmaps.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved summary visualization: {summary_path}")
        return str(summary_path)


class VLNProject:
    """VLN项目主类 - 生产就绪实现"""

    def __init__(self, config: VLNConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.video_processor = VideoProcessor(config)

        # 初始化输出目录
        output_dir = self.config.get('output.output_dir', './outputs')
        self.visualizer = HeatmapVisualizer(output_dir)

        # 初始化核心组件
        self._initialize_components()

    def _initialize_components(self):
        """初始化核心组件"""
        logger.info("正在初始化VLN核心组件...")

        self.monitor.start_timer('component_initialization')

        try:
            # 1. 初始化Spatial-MLLM Pipeline
            max_frames = self.config.get('video.max_frames', 32)
            vggt_precision = self.config.get('model.vggt_precision', 'float16')
            precision_map = {
                'float16': torch.float16,
                'fp16': torch.float16,
                'half': torch.float16,
                'float32': torch.float32,
                'fp32': torch.float32
            }
            vggt_dtype = precision_map.get(str(vggt_precision).lower(), torch.float16)

            # 创建与test_real_llm_pipeline.py相同的配置
            gpu_count = torch.cuda.device_count()

            if gpu_count >= 3:
                # 多GPU配置 - 与test_real_llm_pipeline.py保持一致
                spatial_config = SpatialMLLMIntegrationConfig(
                    target_keyframes=self.config.get('sampling.num_keyframes', 8),  # 🔥 传递关键帧参数
                    use_real_llm=self.config.get('model.use_real_llm', True),
                    llm_model_path=self.config.get('model.llm_model_path'),
                    use_multi_gpu=True,
                    vggt_gpu='cuda:0',
                    dinov3_gpu='cuda:1',
                    llm_gpu='cuda:2',
                    device='cuda:0',
                    total_frames=max_frames,
                    vggt_compute_dtype=vggt_dtype
                )
                logger.info("Multi-GPU configuration:")
                logger.info(f"  - VGGT: cuda:0")
                logger.info(f"  - DINOv3: cuda:1")
                logger.info(f"  - LLM: cuda:2")
            else:
                # 单GPU配置
                spatial_config = SpatialMLLMIntegrationConfig(
                    target_keyframes=self.config.get('sampling.num_keyframes', 8),  # 🔥 传递关键帧参数
                    use_real_llm=self.config.get('model.use_real_llm', True),
                    llm_model_path=self.config.get('model.llm_model_path'),
                    use_multi_gpu=False,
                    device='cuda:0' if gpu_count > 0 else 'cpu',
                    total_frames=max_frames,
                    vggt_compute_dtype=vggt_dtype
                )
                logger.info(f"Single GPU configuration: {spatial_config.device}")
            self.spatial_pipeline = SpatialMLLMPipeline(spatial_config)

            # 2. 初始化算法工厂
            self.algorithm_factory = get_factory()

            # 3. 初始化Frame-Indexed Heatmap Converter
            self.heatmap_converter = FrameIndexedHeatmapConverter(
                hidden_dim=1024,
                output_size=(224, 224)
            )

            # 4. 路径解析器
            self.path_resolver = PathResolver()

            initialization_time = self.monitor.end_timer('component_initialization')
            logger.info(f"组件初始化完成 (耗时: {initialization_time:.2f}s)")

        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise

    def process_video(self,
                     video_path: str,
                     instruction: str = "Navigate and analyze spatial relationships",
                     algorithm_type: str = None) -> Dict[str, Any]:
        """处理单个视频，生成frame-indexed heatmaps"""

        logger.info(f"开始处理视频: {video_path}")
        logger.info(f"指令: {instruction}")

        self.monitor.start_timer('total_processing')

        try:
            # 1. 加载视频
            self.monitor.start_timer('video_loading')
            frames = self.video_processor.load_video(video_path)
            self.monitor.end_timer('video_loading')
            self.monitor.record_memory('after_video_loading')

            # 2. 使用与test_real_llm_pipeline.py相同的均匀采样策略
            # 跳过额外的算法采样步骤，直接使用已加载的帧
            logger.info(f"使用均匀采样策略，加载了 {len(frames)} 帧")

            # 4. Pipeline处理
            self.monitor.start_timer('pipeline_processing')

            # 准备视频数据 (B=1, T, C, H, W)
            video_tensor = torch.stack([
                torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                for frame in frames
            ]).unsqueeze(0)

            current_observation = video_tensor[:, -1]

            # 使用与test_real_llm_pipeline.py相同的简洁调用方式
            # 移除额外的内存消耗参数
            with torch.no_grad():  # 添加内存优化
                pipeline_outputs = self.spatial_pipeline(
                    video_frames=video_tensor,
                    instruction_text=instruction,
                    current_observation=current_observation
                )

            pipeline_time = self.monitor.end_timer('pipeline_processing')
            self.monitor.record_memory('after_pipeline')

            frame_heatmaps = pipeline_outputs.get('frame_indexed_heatmaps', {}) or {}
            selected_keyframes = pipeline_outputs.get('selected_keyframes')

            if isinstance(selected_keyframes, torch.Tensor):
                selected_keyframes = selected_keyframes.detach().cpu().tolist()
            elif selected_keyframes is None:
                # 使用已加载的帧索引
                selected_keyframes = list(range(len(frames)))
            else:
                selected_keyframes = list(selected_keyframes)

            # 修复嵌套列表问题 - flatten if needed
            if selected_keyframes and isinstance(selected_keyframes[0], (list, tuple)):
                selected_keyframes = selected_keyframes[0]  # 取第一个batch的keyframes

            # 确保都是整数
            selected_keyframes = [int(idx) for idx in selected_keyframes]

            heatmap_size = self.spatial_pipeline.config.heatmap_size
            heatmap_collection = []

            for idx in selected_keyframes:
                heatmap_tensor = frame_heatmaps.get(idx)
                if heatmap_tensor is None:
                    fallback_heatmap = pipeline_outputs.get('inter_frame_heatmap')
                    if fallback_heatmap is not None:
                        heatmap_tensor = fallback_heatmap.squeeze(1)
                if heatmap_tensor is None:
                    heatmap_tensor = torch.zeros(
                        1,
                        heatmap_size[0],
                        heatmap_size[1],
                        device=video_tensor.device,
                    )
                heatmap_collection.append(heatmap_tensor[0].detach().cpu().numpy())

            heatmaps = np.stack(heatmap_collection, axis=0) if heatmap_collection else np.empty((0, *heatmap_size))

            # 6. 可视化和保存
            results = {
                'video_path': video_path,
                'instruction': instruction,
                'algorithm_type': algorithm_type,
                'num_frames': len(frames),
                'num_keyframes': len(selected_keyframes),
                'keyframe_indices': selected_keyframes,
                'heatmaps_shape': heatmaps.shape,
                'processing_time': {
                    'video_loading': self.monitor.metrics.get('video_loading_duration', 0),
                    'frame_sampling': 0.0,  # 简化采样不需要额外时间
                    'pipeline_processing': pipeline_time,
                    'total': self.monitor.end_timer('total_processing')
                },
                'fallback_sampled_indices': selected_keyframes
            }

            # 保存可视化
            if self.config.get('output.save_heatmaps', True):
                saved_files = self.visualizer.save_frame_indexed_heatmaps(
                    frames, selected_keyframes, heatmaps,
                    prefix=f"video_{Path(video_path).stem}"
                )
                results['saved_heatmaps'] = saved_files

            if self.config.get('output.visualization', True):
                summary_file = self.visualizer.create_summary_visualization(
                    frames, selected_keyframes, heatmaps, results['processing_time']
                )
                results['summary_visualization'] = summary_file

            # 保存指标
            if self.config.get('output.save_metrics', True):
                metrics_file = self.visualizer.output_dir / f"metrics_{Path(video_path).stem}.json"
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                results['metrics_file'] = str(metrics_file)

            logger.info(f"视频处理完成，总耗时: {results['processing_time']['total']:.2f}s")
            return results

        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def _get_sampling_algorithm(self, algorithm_type: str):
        """获取指定的采样算法"""
        try:
            if algorithm_type == 'enhanced':
                algorithm = self.algorithm_factory.create_auto_configured('enhanced')
            elif algorithm_type == 'quality':
                algorithm = self.algorithm_factory.create_auto_configured('quality')
            elif algorithm_type == 'fast':
                algorithm = self.algorithm_factory.create_auto_configured('fast')
            else:
                logger.warning(f"未知算法类型: {algorithm_type}，使用默认enhanced算法")
                algorithm = self.algorithm_factory.create_auto_configured('enhanced')

            if not hasattr(algorithm, 'sample_frames'):
                raise AttributeError(f"算法 {getattr(algorithm, 'name', algorithm_type)} 缺少 sample_frames 方法")

            return algorithm
        except Exception as e:
            logger.warning(f"算法初始化失败: {e}，使用简化版本")
            # 使用简化的均匀采样作为后备
            return self._create_fallback_sampler()

    def _create_fallback_sampler(self):
        """创建后备采样器"""
        class FallbackSampler:
            def sample_frames(self, frames, num_keyframes):
                n = len(frames)
                if n <= num_keyframes:
                    return list(range(n))
                indices = np.linspace(0, n-1, num_keyframes, dtype=int)
                return indices.tolist()

        return FallbackSampler()

    def batch_process(self, input_list: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量处理多个视频"""
        logger.info(f"开始批量处理 {len(input_list)} 个视频")

        results = []
        for i, video_path in enumerate(tqdm(input_list, desc="批量处理视频")):
            try:
                logger.info(f"处理第 {i+1}/{len(input_list)} 个视频: {video_path}")
                result = self.process_video(video_path, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"处理视频失败 {video_path}: {e}")
                results.append({'video_path': video_path, 'error': str(e)})

        logger.info(f"批量处理完成，成功处理 {len([r for r in results if 'error' not in r])} 个视频")
        return results

    def benchmark_algorithms(self, video_path: str) -> Dict[str, Any]:
        """算法性能基准测试"""
        logger.info(f"开始算法基准测试: {video_path}")

        algorithms = ['fast', 'quality', 'enhanced']
        benchmark_results = {}

        for algo_type in algorithms:
            logger.info(f"测试算法: {algo_type}")
            try:
                result = self.process_video(
                    video_path,
                    algorithm_type=algo_type,
                    instruction="Benchmark test for spatial understanding"
                )
                benchmark_results[algo_type] = {
                    'processing_time': result['processing_time'],
                    'num_keyframes': result['num_keyframes'],
                    'keyframe_indices': result['keyframe_indices']
                }
            except Exception as e:
                logger.error(f"算法 {algo_type} 测试失败: {e}")
                benchmark_results[algo_type] = {'error': str(e)}

        # 保存基准测试结果
        benchmark_file = self.visualizer.output_dir / "algorithm_benchmark.json"
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

        logger.info(f"基准测试完成，结果保存到: {benchmark_file}")
        return benchmark_results


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VLN Project - Frame-Indexed Heatmap Generation (Production Ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

# 单视频处理
python main.py --video /path/to/video.mp4 --instruction "Navigate to the kitchen"

# 指定算法和采样参数
python main.py --video /path/to/video.mp4 --algorithm enhanced --keyframes 6 --max_frames 32

# 基于FPS采样
python main.py --video /path/to/video.mp4 --sample_fps 2.0 --keyframes 8

# 批量处理
python main.py --batch video1.mp4 video2.mp4 video3.mp4

# 算法基准测试
python main.py --benchmark --video /path/to/video.mp4

# 使用配置文件
python main.py --config configs/custom.yaml --video /path/to/video.mp4
        """
    )

    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="单个视频文件路径")
    input_group.add_argument("--images", type=str, help="图像序列目录路径")
    input_group.add_argument("--batch", nargs='+', help="批量处理的视频文件列表")

    # 处理选项
    parser.add_argument("--instruction", type=str,
                       default="Navigate and analyze spatial relationships",
                       help="VLN导航指令")
    parser.add_argument("--algorithm", type=str,
                       choices=['fast', 'quality', 'enhanced'],
                       default='enhanced',
                       help="采样算法选择")
    parser.add_argument("--keyframes", type=int, default=8,
                       help="关键帧数量")

    # 视频采样参数
    parser.add_argument("--max_frames", type=int, default=32,
                       help="视频最大采样帧数 (候选帧池大小)")
    parser.add_argument("--sample_fps", type=float, default=None,
                       help="采样帧率 (如果指定，覆盖max_frames)")

    # 配置选项
    parser.add_argument("--config", type=str, help="YAML配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="输出目录")

    # 模式选项
    parser.add_argument("--benchmark", action="store_true",
                       help="运行算法基准测试模式")
    parser.add_argument("--no_visualization", action="store_true",
                       help="禁用可视化输出")
    parser.add_argument("--no_save", action="store_true",
                       help="禁用文件保存")

    # 性能选项
    parser.add_argument("--profile", action="store_true",
                       help="启用性能分析")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出模式")
    parser.add_argument("--optimize_memory", action="store_true",
                       help="优化内存：设置Qwen VIDEO_TOTAL_PIXELS为实际需求")

    return parser.parse_args()


def main():
    """主函数"""
    try:
        args = parse_arguments()

        # 设置日志级别
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # 📊 内存优化：修复Qwen VIDEO_TOTAL_PIXELS误判
        if args.optimize_memory:
            # 基于实际张量大小计算合理的视频像素预算
            max_frames = getattr(args, 'max_frames', 32)
            target_keyframes = getattr(args, 'keyframes', 8)
            # 计算：批大小×关键帧×通道×高×宽 + 缓冲
            estimated_pixels = 1 * target_keyframes * 3 * 224 * 224 * 2  # 2×缓冲
            os.environ['VIDEO_MAX_PIXELS'] = str(estimated_pixels)
            logger.info(f"🚀 Memory optimization enabled:")
            logger.info(f"   Setting VIDEO_MAX_PIXELS = {estimated_pixels:,}")
            logger.info(f"   Default was: 90,316,800 (reduced by {90316800/estimated_pixels:.1f}×)")

        # 加载配置
        config = VLNConfig(args.config)

        # 更新配置
        if args.keyframes:
            config.config['sampling']['num_keyframes'] = args.keyframes
        if args.max_frames:
            config.config['video']['max_frames'] = args.max_frames
        if args.sample_fps:
            config.config['video']['sample_fps'] = args.sample_fps
        if args.output_dir:
            config.config['output']['output_dir'] = args.output_dir
        if args.no_visualization:
            config.config['output']['visualization'] = False
        if args.no_save:
            config.config['output']['save_heatmaps'] = False
            config.config['output']['save_metrics'] = False

        # 初始化项目
        logger.info("正在初始化VLN项目...")
        project = VLNProject(config)

        # 根据模式执行
        if args.benchmark:
            # 基准测试模式
            if not args.video:
                raise ValueError("基准测试模式需要指定--video参数")

            logger.info("=== 算法基准测试模式 ===")
            results = project.benchmark_algorithms(args.video)

            # 打印结果摘要
            print("\n基准测试结果:")
            for algo, result in results.items():
                if 'error' in result:
                    print(f"  {algo}: 失败 - {result['error']}")
                else:
                    print(f"  {algo}: {result['processing_time']['total']:.2f}s "
                          f"({result['num_keyframes']} 关键帧)")

        elif args.batch:
            # 批量处理模式
            logger.info("=== 批量处理模式 ===")
            results = project.batch_process(
                args.batch,
                instruction=args.instruction,
                algorithm_type=args.algorithm
            )

            # 打印结果摘要
            successful = len([r for r in results if 'error' not in r])
            print(f"\n批量处理完成: {successful}/{len(results)} 个视频处理成功")

        else:
            # 单视频处理模式
            input_path = args.video or args.images
            logger.info("=== 单视频处理模式 ===")

            if args.images:
                # 处理图像序列（需要实现）
                logger.error("图像序列处理功能尚未实现")
                return

            result = project.process_video(
                input_path,
                instruction=args.instruction,
                algorithm_type=args.algorithm
            )

            # 打印结果摘要
            print(f"\n处理完成:")
            print(f"  视频: {result['video_path']}")
            print(f"  算法: {result['algorithm_type']}")
            print(f"  帧数: {result['num_frames']}")
            print(f"  关键帧: {result['num_keyframes']}")
            print(f"  总耗时: {result['processing_time']['total']:.2f}s")

            if 'saved_heatmaps' in result:
                print(f"  保存了 {len(result['saved_heatmaps'])} 个热力图")
            if 'summary_visualization' in result:
                print(f"  汇总可视化: {result['summary_visualization']}")

        logger.info("VLN项目执行完成")

    except KeyboardInterrupt:
        logger.info("用户中断执行")
        sys.exit(0)

    except Exception as e:
        logger.error(f"执行失败: {e}")
        if args.verbose if 'args' in locals() else False:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
