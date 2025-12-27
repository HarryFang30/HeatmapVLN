#!/usr/bin/env python3
"""
验证 STOP 样本采样修复效果

检查修复后 STOP 样本的比例是否提升到预期水平（~2.5%）
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import yaml
from tqdm import tqdm
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset

def verify_stop_sampling(config_path: str):
    """验证 STOP 采样修复效果"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    data_cfg = cfg['data']
    sw_cfg = data_cfg['sliding_window']
    
    print("=" * 60)
    print("STOP 样本采样验证")
    print("=" * 60)
    print(f"数据集根目录: {data_cfg['root']}")
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
    
    # 统计 STOP 样本比例
    print("正在统计 STOP 样本比例（采样 2000 个样本）...")
    sample_size = min(2000, len(train_dataset))
    
    stop_count = 0
    valid_count = 0
    
    import random
    random.seed(42)
    sampled_indices = random.sample(range(len(train_dataset)), sample_size)
    
    for idx in tqdm(sampled_indices):
        try:
            sample = train_dataset[idx]
            if sample['action_valid'] > 0.5:
                valid_count += 1
                if sample['is_stop'] > 0.5:
                    stop_count += 1
        except Exception as e:
            print(f"Warning: Failed to load sample {idx}: {e}")
            continue
    
    print()
    print("=" * 60)
    print("统计结果")
    print("=" * 60)
    
    if valid_count > 0:
        stop_ratio = stop_count / valid_count
        print(f"有效样本数: {valid_count}")
        print(f"STOP 样本数: {stop_count}")
        print(f"STOP 比例: {stop_ratio*100:.2f}%")
        print()
        
        # 评估修复效果
        expected_min = 2.0  # 期望最少 2%
        expected_max = 3.5  # 期望最多 3.5%
        
        if stop_ratio * 100 < expected_min:
            print(f"⚠️  警告: STOP 比例过低（< {expected_min}%）")
            print(f"   预期: {expected_min}% - {expected_max}%")
            print(f"   实际: {stop_ratio*100:.2f}%")
            print(f"   建议: 检查数据集采样逻辑是否正确应用")
        elif stop_ratio * 100 > expected_max:
            print(f"⚠️  警告: STOP 比例过高（> {expected_max}%）")
            print(f"   预期: {expected_min}% - {expected_max}%")
            print(f"   实际: {stop_ratio*100:.2f}%")
        else:
            print(f"✅ STOP 比例在预期范围内（{expected_min}% - {expected_max}%）")
            print(f"   实际: {stop_ratio*100:.2f}%")
            print(f"   修复成功！")
    else:
        print("❌ 错误: 没有找到有效样本")
    
    print()
    print("验证完成！")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="验证 STOP 样本采样修复效果")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    verify_stop_sampling(args.config)

