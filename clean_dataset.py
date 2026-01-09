#!/usr/bin/env python
# coding=utf-8
"""
清洗数据集脚本（多进程 + 低内存）
过滤掉所有 mask 均为 None 的样本，并清理 obj2mask 中的 None 条目
"""

from datasets import load_from_disk
import logging
import os
import numpy as np
# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def mask2grid(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    grid_size = 24
    target_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    mask_height, mask_width = mask.shape

    cell_height = mask_height / grid_size
    cell_width = mask_width / grid_size

    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            mask_y_start = int(grid_y * cell_height)
            mask_y_end = int((grid_y + 1) * cell_height)
            mask_x_start = int(grid_x * cell_width)
            mask_x_end = int((grid_x + 1) * cell_width)

            mask_y_start = max(0, mask_y_start)
            mask_y_end = min(mask_height, mask_y_end)
            mask_x_start = max(0, mask_x_start)
            mask_x_end = min(mask_width, mask_x_end)

            cell_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            total_pixels = cell_mask.size
            ones_count = np.sum(cell_mask)

            if total_pixels > 0 and ones_count / total_pixels > 0.5:
                target_grid[grid_y, grid_x] = 1.0

    return target_grid


def clean_sample(sample):
    """
    清洗单个样本：
    - 移除 obj2mask 中值为 None 的项
    - 如果清洗后为空，则整个样本应被过滤掉（返回 None）
    """
    new_sample = dict(sample) 
    obj2mask = sample.get("obj2mask", {})
    if not isinstance(obj2mask, dict):
        return None  # 非法格式，过滤

    # 仅保留非 None 的 mask
    cleaned = {k: v for k, v in obj2mask.items() if v is not None}
    
    if not cleaned:
        print('error!')
        return None  # 全是 None，过滤掉
    
    new_sample["obj2mask"] = cleaned
    obj2grid = {k:mask2grid(v) for k,v in cleaned.items()}
    new_sample["obj2grid"] = obj2grid
    return new_sample

def main():
    dataset_path = "/root/autodl-tmp/dataset/OHD-Caps-train-sam3"
    cleaned_dataset_path = "/root/autodl-tmp/dataset/OHD-Caps-train-sam3-cleaned"

    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return

    logger.info(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    logger.info(f"过滤前数据集大小: {len(dataset)}")

    # 第一步：清洗并标记无效样本为 None
    logger.info("开始并行清洗样本（多进程）...")
    cleaned_dataset = dataset.map(
        clean_sample,
        num_proc=20,      # 使用全部 CPU 核心
        desc="清洗样本",
        load_from_cache_file=False,
        writer_batch_size=100,       # 控制写入批次大小，节省内存
        keep_in_memory=False          # 不强制保留在内存
    )

    # 第二步：过滤掉返回 None 的样本
    # logger.info("过滤无效样本...")
    # final_dataset = cleaned_dataset.filter(
    #     lambda x: x is not None and x["obj2mask"],  # 确保不为 None 且非空
    #     num_proc=16,
    #     desc="过滤样本"
    # )

    # logger.info(f"过滤后数据集大小: {len(final_dataset)}")
    logger.info(f"正在保存清洗后的数据集: {cleaned_dataset_path}")
    cleaned_dataset.save_to_disk(cleaned_dataset_path)
    logger.info("✅ 数据集清洗完成！")

if __name__ == "__main__":
    main()