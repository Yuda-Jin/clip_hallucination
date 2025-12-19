import json
import os
import sys
from datasets import Dataset, Image
from PIL import Image as PILImage
import pandas as pd
from tqdm import tqdm

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

def build_hf_dataset(json_file_path, image_dir=None, detector=None, batch_size=1000):
    # 设置默认图像目录
    if image_dir is None:
        # 假设图像在data/OHD-Caps/train/images目录下
        base_dir = os.path.dirname(json_file_path)
        image_dir = os.path.join(base_dir, 'images')
    
    print(f"正在读取JSON文件: {json_file_path}")
    print(f"图像目录: {image_dir}")
    
    # 读取JSON文件 - 使用生成器避免一次性加载所有数据
    def json_generator():
        for file_path in json_file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"解析JSON行时出错: {e}")
                            continue
    
    # 分批处理数据
    print("开始分批处理数据...")
    total_processed = 0
    total_miss = 0
    total_repit = 0
    batches = []
    current_batch = []
    
    # 遍历所有数据
    for idx, item in enumerate(tqdm(json_generator())):
        # 获取图像文件名
        image_filename = item['file_path']
        # 构建完整的图像路径
        image_path = os.path.join(image_dir, image_filename)
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 - {image_path}")
            continue
        
        # 尝试打开图像以验证其有效性
        try:
            # 只验证图像有效性，不存储完整图像对象
            with PILImage.open(image_path) as img:
                # 转换为RGB格式以验证兼容性
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            # 关闭图像文件，释放内存
        except Exception as e:
            print(f"警告: 无效的图像文件 - {image_path}, 错误: {e}")
            continue
        
        # 为数据集准备数据项，只保留必要的字段
        dataset_item = {
            'image_filename': image_filename,  # 图像文件名
            'image_path': image_path,  # 完整图像路径
            'image': image_path,  # 只存储路径，让datasets.Image自动加载
            'positive_sample': item['positive_sample'],  # 保留原有的positive_sample
            'ground_truth': item['ground_truth'],  # 保留原有的ground_truth
            'popular': item['popular'],  # 保留原有的popular
            'random': item['random'],  # 保留原有的random
            'adversarial': item['adversarial'],  # 保留原有的adversarial
        }
        
        # 计算hallucinated对象
        other_obj = set(item['adversarial'] + item['popular'] + item['random'])
        hal_obj = []
        for obj in other_obj:
            if obj not in item['ground_truth']:
                hal_obj.append(obj)
        dataset_item['hal_obj'] = hal_obj
        
        # 进行物体检测
        detections = []
        if detector is not None:
            _, detections = detector.grounded_segmentation(
                image=image_path,
                labels=item['ground_truth'],
                polygon_refinement=True,
            )
        
        # 处理检测结果
        label = []
        result = []
        for detection in detections:
            if detection.score >= 0.4 or detection.label not in label:
                detection.grid = mask2grid(detection.mask)
                # 将DetectionResult对象转换为字典，只保留必要的字段
                detection_dict = {
                    'label': detection.label,
                    'score': detection.score,
                    'grid': detection.grid
                }
                result.append(detection_dict)
                label.append(detection.label)
        
        dataset_item['mask'] = result
        
        # 计算统计信息
        detected = set([detection['label'] for detection in result])  # 检测到的对象
        transfered = [label if label.endswith(".") else label + "." for label in item['ground_truth']]  # GT 对象
        miss = len([obj for obj in transfered if obj not in detected])  # 没有检测到的对象
        repit = len(result) - len(item['ground_truth']) + miss
        total_miss += miss
        total_repit += repit
        
        # 添加到当前批次
        current_batch.append(dataset_item)
        total_processed += 1
        
        # 当批次达到指定大小时，保存批次
        if len(current_batch) >= batch_size:
            print(f"正在保存批次，包含 {len(current_batch)} 条数据")
            # 创建批次数据集
            batch_dataset = Dataset.from_list(current_batch)
            try:
                batch_dataset = batch_dataset.cast_column("image", Image())
            except Exception as e:
                print(f"警告: 转换图像列时出错，但数据集仍可使用 - {e}")
            
            batches.append(batch_dataset)
            # 清空当前批次以释放内存
            current_batch = []
    
    # 处理剩余的数据
    if current_batch:
        print(f"正在保存最后一批次，包含 {len(current_batch)} 条数据")
        batch_dataset = Dataset.from_list(current_batch)
        try:
            batch_dataset = batch_dataset.cast_column("image", Image())
        except Exception as e:
            print(f"警告: 转换图像列时出错，但数据集仍可使用 - {e}")
        
        batches.append(batch_dataset)
    
    print(f"成功处理 {total_processed} 条有效数据")
    print(f"总缺失: {total_miss}, 总重复: {total_repit}")
    
    # 合并所有批次
    from datasets import concatenate_datasets
    if batches:
        print(f"正在合并 {len(batches)} 个批次...")
        dataset = concatenate_datasets(batches)
    else:
        print("没有有效数据可处理")
        return None
    
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
    detector = detector_seg(threshold = 0.2)
    dataset = build_hf_dataset(json_file_path, image_dir,detector)


    # 保存合并后的数据集
    output_dir = '/root/autodl-tmp/dataset/OHD-Caps-train-mask'
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