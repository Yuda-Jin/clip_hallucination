#!/usr/bin/env python
# coding=utf-8
"""
清洗数据集脚本
过滤掉所有 mask 均为 None 的样本
"""

from datasets import load_from_disk, Dataset
import logging
import os
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def filter_none_masks(sample):
    """过滤掉所有 mask 均为 None 的样本"""
    try:
        # 检查 sample 是否为字典
        if not isinstance(sample, dict):
            logger.error(f"样本不是字典，类型为: {type(sample)}")
            return False
        
        # 检查 obj2mask 是否存在
        if 'obj2mask' not in sample:
            logger.warning(f"样本中不存在 obj2mask 字段")
            return False
        
        obj2mask = sample['obj2mask']
        
        # 检查 obj2mask 的类型
        if isinstance(obj2mask, str):
            logger.error(f"obj2mask 是字符串，不是字典或列表")
            return False
        
        # 检查 obj2mask 是否为空
        if not obj2mask:
            logger.info(f"obj2mask 为空")
            return False
        
        # 检查 obj2mask 中是否存在至少一个非 None 的 mask
        if isinstance(obj2mask, dict):
            has_valid_mask = any(mask is not None for mask in obj2mask.values())
        else:
            has_valid_mask = any(mask is not None for mask in obj2mask)
        
        return has_valid_mask
    except Exception as e:
        logger.error(f"过滤样本时出错: {e}")
        return False

def main():
    # 数据集路径
    dataset_path = "/root/autodl-tmp/dataset/OHD-Caps-train-sam3"
    # 清洗后的数据集路径
    cleaned_dataset_path = "/root/autodl-tmp/dataset/OHD-Caps-train-sam3-cleaned"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return
    
    # 加载数据集
    logger.info(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # 过滤数据集
    logger.info(f"过滤前数据集大小: {len(dataset)}")
    
    # 使用 for 循环过滤
    cleaned_samples = []
    for sample in tqdm(dataset, desc="过滤数据集", unit="样本"):

        # 检查 obj2mask 中是否存在至少一个非 None 的 mask
        if any(mask is not None for mask in sample['obj2mask'].values()):
            sample.update({'obj2mask': {k: v for k, v in sample['obj2mask'].items() if v is not None}})
            cleaned_samples.append(sample)
        else:
            print("find error")
            
    
    # 创建新的数据集
    dataset = Dataset.from_list(cleaned_samples)
    logger.info(f"过滤后数据集大小: {len(dataset)}")
    
    # 保存清洗后的数据集
    logger.info(f"正在保存清洗后的数据集: {cleaned_dataset_path}")
    dataset.save_to_disk(cleaned_dataset_path)
    
    logger.info(f"数据集清洗完成")

if __name__ == "__main__":
    main()