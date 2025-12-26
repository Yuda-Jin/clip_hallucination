#!/usr/bin/env python
# coding=utf-8
"""
清洗数据集脚本（多进程 + 低内存）
过滤掉所有 mask 均为 None 的样本，并清理 obj2mask 中的 None 条目
"""

from datasets import load_from_disk
import logging
import os

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clean_sample(sample):
    """
    清洗单个样本：
    - 移除 obj2mask 中值为 None 的项
    - 如果清洗后为空，则整个样本应被过滤掉（返回 None）
    """
    obj2mask = sample.get("obj2mask", {})
    if not isinstance(obj2mask, dict):
        return None  # 非法格式，过滤

    # 仅保留非 None 的 mask
    cleaned = {k: v for k, v in obj2mask.items() if v is not None}
    
    if not cleaned:
        print('error!')
        return None  # 全是 None，过滤掉
    
    sample["obj2mask"] = cleaned
    return sample

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