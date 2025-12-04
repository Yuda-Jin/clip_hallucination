import json
import os
import sys
from datasets import Dataset, Image
from PIL import Image as PILImage
import pandas as pd
from tqdm import tqdm
import torch
#################################### For Image ####################################
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
custom_path = "../"
import numpy as np
if custom_path not in sys.path:
    sys.path.insert(0, custom_path)  # 插入开头，优先级最高
from grounded_sam_utils import detector_seg

def mask2grid(mask):
    # 初始化24*24网格
    grid_size = 24
    target_grid = np.zeros((grid_size, grid_size), dtype=int)
    # 获取掩码尺寸
    mask_height, mask_width = mask.shape
    binary_mask = (mask == 255).astype(np.uint8)
    # 计算每个网格单元格在原始掩码中的大小
    cell_height = mask_height / grid_size
    cell_width = mask_width / grid_size
    
    # 对于每个网格单元格，判断是否包含掩码像素
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            # 计算当前网格单元格在原始掩码中的坐标范围
            mask_y_start = int(grid_y * cell_height)
            mask_y_end = int((grid_y + 1) * cell_height)
            mask_x_start = int(grid_x * cell_width)
            mask_x_end = int((grid_x + 1) * cell_width)
            
            # 确保坐标在有效范围内
            mask_y_start = max(0, mask_y_start)
            mask_y_end = min(mask_height, mask_y_end)
            mask_x_start = max(0, mask_x_start)
            mask_x_end = min(mask_width, mask_x_end)
            
            # 检查当前单元格内1的像素是否占绝大多数（超过50%）
            cell_mask = binary_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            total_pixels = cell_mask.size
            ones_count = np.sum(cell_mask)
            
            # 如果1的像素占比超过50%，则将网格单元格设为1
            if total_pixels > 0 and ones_count / total_pixels > 0.5:
                target_grid[grid_y, grid_x] = 1
    
    return target_grid

def build_hf_dataset(json_file_path, image_dir=None,detector=None):
    sam3_model = build_sam3_image_model(   
        bpe_path=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        checkpoint_path='/root/autodl-tmp/cache/modelscope/models/facebook/sam3/sam3.pt',
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False)
    sam3_processor = Sam3Processor(sam3_model)
    # 设置默认图像目录
    if image_dir is None:
        # 假设图像在data/OHD-Caps/train/images目录下
        base_dir = os.path.dirname(json_file_path)
        image_dir = os.path.join(base_dir, 'images')
    
    print(f"正在读取JSON文件: {json_file_path}")
    print(f"图像目录: {image_dir}")
    
    # 读取JSON文件
    data = []
    for file_path in json_file_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"解析JSON行时出错: {e}")
                        continue
    
    print(f"成功读取 {len(data)} 条数据")
    
    # 准备数据集数据
    dataset_data = []
    qbar = tqdm(enumerate(data))
    total_miss = 0
    total_repit = 0
    for idx,item in qbar:   
        # 获取图像文件名
        image_filename = item['file_path']
        # 构建完整的图像路径
        image_path = os.path.join(image_dir, image_filename)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 - {image_path}")
            # 跳过不存在的图像
            continue
        
        # 尝试打开图像以验证其有效性

        img = PILImage.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 为数据集准备数据项
        dataset_item = {
            'image_filename': image_filename,  # 图像路径，将由datasets.Image自动处理
            'image_path': image_path,
            'image':img,
            'ground_truth':item['ground_truth'],
            'adversarial':item['adversarial'],
            'popular':item['popular'],
            'random':item['random']
        }
        other_obj = set(item['adversarial']+item['popular']+item['random'])
        hal_obj = []
        for obj in other_obj:
            if obj not in item['ground_truth']:
                hal_obj.append(obj)
        dataset_item.update({'hal_obj':hal_obj})
        obj2mask = {}
        obj2grid = {}
        miss = 0
        inference_state = sam3_processor.set_image(img)
        for obj in item['ground_truth']:
            output = sam3_processor.set_text_prompt(state=inference_state, prompt=obj)
            if output['masks'].shape[0]==0:
                miss +=1
                H, W = output['masks'].shape[-2:]
                mask = np.full((H, W), False, dtype=bool)
                # obj2mask[obj] = mask
                obj2grid[obj] = mask2grid(mask)
            else:
                mask = output['masks'].squeeze(1).detach().cpu().numpy()
                merge_mask = np.any(mask,axis=0)
                obj2mask[obj] = merge_mask
                obj2grid[obj] = mask2grid(merge_mask)
        total_miss += miss
        
        qbar.set_postfix({'miss':miss,'avg_miss':total_miss/(idx+1)})
        
        dataset_item.update({'obj2mask':obj2mask,'obj2grid':obj2grid})
        dataset_data.append(dataset_item)
    
    print(f"成功处理 {len(dataset_data)} 条有效数据")
    print(f"总缺失: {total_miss}")
    
    # 直接从数据列表创建数据集，绕过DataFrame
    print(f"正在创建数据集，包含 {len(dataset_data)} 条数据")
    dataset = Dataset.from_list(dataset_data)
    
    # 不需要额外转换，因为我们已经存储了PIL图像对象
    # 但为了确保与HuggingFace datasets的兼容性，仍然指定类型
    try:
        dataset = dataset.cast_column("image", Image())
    except Exception as e:
        print(f"警告: 转换图像列时出错，但数据集仍可使用 - {e}")
    
    print("数据集构建完成！")
    print(f"数据集大小: {len(dataset)}")
    print(f"数据集特征: {list(dataset.features.keys())}")
    
    return dataset


def main():
    """
    主函数
    """
    # JSON文件路径
    json_file_path = ['/root/DFLR/clip_hallucination/data/OHD-Caps/train/coco_annotations_8k.json',
                      'data/OHD-Caps/train/flickr_annotations_8k.json'      ]
    image_dir = '/root/autodl-tmp/dataset/coco/train2017'
    dataset = build_hf_dataset(json_file_path, image_dir)


    # 保存合并后的数据集
    output_dir = '/root/autodl-tmp/dataset/OHD-Caps-train-seem'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset.save_to_disk(output_dir)
    print(f"合并后的数据集已保存到: {output_dir}")
    
    # 显示数据集的一些示例
    print("\n数据集示例:")
    print(dataset[0])
    print("\n数据集特征:")
    print(list(dataset.features.keys()))

if __name__ == "__main__":
    main()