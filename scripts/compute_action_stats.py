#!/usr/bin/env python3
"""
计算数据集的实际动作统计值

用于确定 action_head 的正确归一化范围。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from tqdm import tqdm
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset

def compute_dataset_action_stats(config_path: str, margin: float = 0.1):
    """
    遍历数据集计算实际动作统计值
    
    Args:
        config_path: 配置文件路径
        margin: 添加的安全余量（例如 0.1 = 10%）
    """
    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg['data']
    sw_cfg = data_cfg['sliding_window']
    
    print("=" * 60)
    print("数据集动作统计分析")
    print("=" * 60)
    print(f"数据集根目录: {data_cfg['root']}")
    print(f"图像尺寸: {data_cfg['image_size']}")
    print(f"采样步长: {sw_cfg.get('sample_stride', 1)}")
    print()
    
    # 创建训练集
    print("正在加载训练集...")
    train_dataset = VLNSlidingWindowDataset(
        root=data_cfg['root'],
        split='train',
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(data_cfg['image_size']),
        hm_size=tuple(data_cfg['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
        sample_stride=sw_cfg.get('sample_stride', 1),
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print()
    
    # 使用数据集的内置方法计算统计值
    print("正在计算动作统计值（这可能需要几分钟）...")
    min_vals, max_vals = train_dataset.compute_action_stats(margin=margin)
    
    print()
    print("=" * 60)
    print("计算结果")
    print("=" * 60)
    print(f"原始范围:")
    print(f"  min: [{min_vals[0]:.4f}, {min_vals[1]:.4f}]")
    print(f"  max: [{max_vals[0]:.4f}, {max_vals[1]:.4f}]")
    print()
    print(f"建议配置（添加 {margin*100:.0f}% 余量）:")
    print(f"  action_stats_min: [{min_vals[0]:.4f}, {min_vals[1]:.4f}]")
    print(f"  action_stats_max: [{max_vals[0]:.4f}, {max_vals[1]:.4f}]")
    print()
    
    # 当前配置
    current_min = cfg.get('model', {}).get('action_head', {}).get('action_stats_min', [-0.5, -0.2])
    current_max = cfg.get('model', {}).get('action_head', {}).get('action_stats_max', [0.5, 1.0])
    print(f"当前配置:")
    print(f"  action_stats_min: {current_min}")
    print(f"  action_stats_max: {current_max}")
    print()
    
    # 检查是否需要更新
    needs_update = False
    for i in range(2):
        if min_vals[i] < current_min[i] or max_vals[i] > current_max[i]:
            needs_update = True
            break
    
    if needs_update:
        print("⚠️  警告: 当前配置的范围不足以覆盖实际数据！")
        print("   建议更新当前所用配置文件中的 action_stats（共享环境默认是 configs/train_config_internnav.yaml）")
    else:
        # 检查是否过大
        range_ratio_0 = (current_max[0] - current_min[0]) / (max_vals[0] - min_vals[0])
        range_ratio_1 = (current_max[1] - current_min[1]) / (max_vals[1] - min_vals[1])
        
        if range_ratio_0 > 3.0 or range_ratio_1 > 3.0:
            print("⚠️  警告: 当前配置的范围过大（是实际范围的 3 倍以上）！")
            print("   这会导致归一化后的动作值接近 0，扩散模型可能学不到有效信号。")
            print("   建议收紧范围到建议值。")
        else:
            print("✅ 当前配置合理。")
    
    print()
    
    # 额外统计：Stop 动作比例
    print("=" * 60)
    print("Stop 动作统计")
    print("=" * 60)
    
    stop_count = 0
    valid_count = 0
    sample_size = min(1000, len(train_dataset))  # 采样 1000 个样本
    
    print(f"采样 {sample_size} 个样本进行统计...")
    for idx in tqdm(np.random.choice(len(train_dataset), sample_size, replace=False)):
        try:
            sample = train_dataset[idx]
            if sample['action_valid'] > 0.5:
                valid_count += 1
                if sample['is_stop'] > 0.5:
                    stop_count += 1
        except:
            continue
    
    if valid_count > 0:
        stop_ratio = stop_count / valid_count
        print(f"STOP 样本比例: {stop_count}/{valid_count} = {stop_ratio*100:.2f}%")
        
        if stop_ratio < 0.01:
            print("⚠️  警告: STOP 样本极度稀疏（< 1%）！")
            print("   模型可能会忽略 STOP 预测，始终输出'不停止'。")
            print("   建议检查 Stop Head 的 Focal Loss 参数是否合理。")
    
    print()
    print("分析完成！")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="计算数据集动作统计值")
    parser.add_argument('--config', type=str, default='configs/train_config_internnav.yaml',
                        help='配置文件路径')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='安全余量（默认 0.1 = 10%）')
    
    args = parser.parse_args()
    
    compute_dataset_action_stats(args.config, args.margin)

