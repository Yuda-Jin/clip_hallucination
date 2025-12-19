#!/usr/bin/env python
# coding=utf-8
"""
数据集预览脚本
用于预览构建好的CLIP数据集，查看样本信息和可视化图像

使用说明：
1. 修改get_fixed_args()函数中的参数值来控制预览行为
2. 直接运行脚本即可：python preview_dataset.py
"""

import os
import random
import numpy as np
from matplotlib import patches, colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from datasets import load_from_disk

# 固定参数设置
def get_fixed_args():
    """
    获取固定的参数值
    """
    class Args:
        def __init__(self):
            # 数据集路径 - 需要根据实际情况修改
            self.dataset_path = "/root/autodl-tmp/dataset/OHD-Caps-train-mask"
            # 要预览的样本数量
            self.num_samples = 10
            # 是否显示mask信息
            self.show_masks = True
            # 是否显示grid信息
            self.show_grids = True
            # 是否随机采样
            self.random_sample = True
    return Args()

def load_dataset(dataset_path):
    """
    加载数据集
    """
    print(f"正在加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"数据集加载完成")
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集特征: {list(dataset.features.keys())}")
    return dataset

def visualize_sample(sample, show_masks=False, show_grids=False,image_filename=None):
    total_cols = 2 + (show_masks and 1) + (show_grids and 1)
    fig = plt.figure(figsize=(20, 10))

    # === 图像 ===
    ax1 = plt.subplot(1, total_cols, 1)
    img = sample["image"]
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif hasattr(img, 'convert'):  # PIL Image
        img = img.convert("RGB")
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    
    ax1.imshow(np.array(img))
    ax1.set_title("Original Image")
    ax1.axis("off")

    # === 信息文本 ===
    ax2 = plt.subplot(1, total_cols, 2)
    ax2.axis("off")
    info_text = [f"{key}: {sample.get(key, 'N/A')}" for key in 
                 ["image_filename", "ground_truth", "adversarial", "popular", "random", "hal_obj"]]
    if "obj" in sample:
        info_text.append(f"Objects: {sample['obj']}")
    text = "\n".join(info_text)
    ax2.text(0, 0.5, text, fontsize=12, va="center", ha="left")
    ax2.set_title("Sample Information")

    # === Mask ===
    if show_masks and "mask" in sample:
        ax3 = plt.subplot(1, total_cols, 3)
        mask_img = np.zeros((24, 24,3), dtype=np.uint8)
        colormap = plt.get_cmap("tab20")

        for i, mask_item in enumerate(sample["mask"]):
            color_float = colormap(i % 20)[:3]  # 归一化 RGB (0~1)
            color_uint8 = (np.array(color_float) * 255).astype(np.uint8)  # 转为 0~255 整数
            for j in range(24):
                for k in range(24):
                    if mask_item["grid"][j][k] > 0:
                        mask_img[j][k] = color_uint8

        ax3.imshow(mask_img)
        ax3.set_title("Object Masks")
        ax3.axis("off")

        patches_list = [
            patches.Patch(color=colormap(i % 20)[:3], label=mask_item["label"])
            for i, mask_item in enumerate(sample["mask"])
        ]
        ax3.legend(handles=patches_list, loc="lower left", bbox_to_anchor=(1.05, 0))

    plt.tight_layout()
    plt.savefig(f'preview_dataset/preview_{image_filename}.png')
    # plt.show()
    
def preview_dataset(dataset, num_samples=5, show_masks=False, show_grids=False, random_sample=False):
    """
    预览数据集
    """
    print(f"\n正在预览 {num_samples} 个样本...")
    
    # 选择样本
    if random_sample:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        indices = range(min(num_samples, len(dataset)))
    
    for i, idx in enumerate(indices):
        print(f"\n=== 样本 {i+1}/{num_samples} (索引: {idx}) ===")
        sample = dataset[idx]
        
        # 显示样本信息
        print(f"图像文件名: {sample['image_filename']}")
        print(f"真实标签: {sample['ground_truth']}")
        print(f"对抗性标签: {sample['adversarial']}")
        print(f"流行标签: {sample['popular']}")
        print(f"随机标签: {sample['random']}")
        print(f"幻觉对象: {sample['hal_obj']}")
        
        if "obj" in sample:
            print(f"对象列表: {sample['obj']}")
        
        if "mask" in sample:
            print(f"检测到的对象数量: {len(sample['mask'])}")
            for mask_item in sample["mask"]:
                print(f"  - {mask_item['label']}: 置信度 {mask_item['score']:.2f}")
        
        # 可视化样本
        visualize_sample(sample, show_masks, show_grids,sample['image_filename'])

def main():
    """
    主函数
    """
    # 使用固定参数
    args = get_fixed_args()
    
    print(f"使用固定参数运行:")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  预览样本数量: {args.num_samples}")
    print(f"  显示mask: {args.show_masks}")
    print(f"  显示grid: {args.show_grids}")
    print(f"  随机采样: {args.random_sample}")
    
    # 加载数据集
    dataset = load_dataset(args.dataset_path)
    
    # 预览数据集
    preview_dataset(
        dataset,
        num_samples=args.num_samples,
        show_masks=args.show_masks,
        show_grids=args.show_grids,
        random_sample=args.random_sample
    )

if __name__ == "__main__":
    main()