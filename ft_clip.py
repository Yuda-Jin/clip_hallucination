#!/usr/bin/env python
# coding=utf-8
"""
CLIP微调脚本
使用Hugging Face Trainer实现CLIP模型的微调
"""

from pathlib import Path
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import transformers
from transformers import (
    CLIPImageProcessor,
    CLIPTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizerBase,
    CLIPProcessor
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# 检查Transformers版本
check_min_version("4.26.0")
require_version("datasets>=1.8.0")

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    模型参数配置
    """
    model_name_or_path: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "预训练模型名称或路径"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "配置文件名称或路径"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "分词器名称或路径"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "特征提取器名称或路径"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "缓存目录"}
    )
    freeze_vision_model: bool = field(
        default=False, metadata={"help": "是否冻结视觉模型"}
    )
    freeze_text_model: bool = field(
        default=False, metadata={"help": "是否冻结文本模型"}
    )


@dataclass
class DataTrainingArguments:
    """
    数据训练参数配置
    """
    dataset_path: str = field(
        metadata={"help": "数据集路径"}
    )
    image_column: str = field(
        default="image", metadata={"help": "图像列名称"}
    )
    caption_column: str = field(
        default="caption", metadata={"help": "文本描述列名称"}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "最大序列长度"}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "最大训练样本数"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "最大评估样本数"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖缓存"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "预处理进程数"}
    )

def mask2grid(mask):
    # 初始化24*24网格
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    grid_size = 24
    target_grid = np.zeros((grid_size, grid_size), dtype=int)
    # 获取掩码尺寸
    mask_height, mask_width = mask.shape
    # binary_mask = (mask == 255).astype(np.uint8)
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
            cell_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            total_pixels = cell_mask.size
            ones_count = np.sum(cell_mask)
            
            # 如果1的像素占比超过50%，则将网格单元格设为1
            if total_pixels > 0 and ones_count / total_pixels > 0.5:
                target_grid[grid_y, grid_x] = 1
    
    return target_grid

class CLIPCollator:
    """
    CLIP数据整理器
    """
    def __init__(self, processor: CLIPProcessor, data_args: DataTrainingArguments):
        self.processor = processor
        self.data_args = data_args
    
    def __call__(self, examples: List[Dict]) -> Dict:
        """
        整理批次数据
        """
        # 分离图像和文本
        images = [example["image"] for example in examples]
        # 给每个obj添加prompt模板
        objs_prompts = []
        obj_counts = []  # 记录每个图像对应的对象数量
        grid_mask = []  # 记录每个图像对应的网格掩码
        
        for example in examples:
            instance_grid = []
            instance_obj_prompt = []
            for obj,mask in example["obj2mask"].items():
                if mask!=None:
                    instance_grid.append(mask2grid(mask))
                    instance_obj_prompt.append(f"There is {obj}.")
            
            objs_prompts.append(instance_obj_prompt)
            obj_counts.append(len(instance_grid))
            grid_mask.append(instance_grid)
        
        captions = [f"{example['positive_sample']}" for example in examples ]
        
        # 使用processor处理图像和文本
        # 分别对captions和objs进行tokenize
        text_inputs = self.processor.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_seq_length,
            return_tensors="pt"
        )
        obj_input_list = []
        for instance_obj_prompt in objs_prompts:
            if instance_obj_prompt:
                obj_inputs = self.processor.tokenizer(
                    instance_obj_prompt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.data_args.max_seq_length,
                    return_tensors="pt"
                )
                obj_input_list.append(obj_inputs)
            else:
                obj_input_list.append(None)

        # 处理图像
        image_inputs = self.processor.image_processor(
            images,
            return_tensors="pt"
        )
        
        # 合并所有输入
        processed_inputs = {
            **text_inputs,
            **image_inputs,
            "obj_input_list": obj_input_list,
            "obj_counts": obj_counts,  # 添加每个图像的对象数量
            "grid_mask": grid_mask     # 添加网格掩码
        }
        # 添加return_loss标志
        processed_inputs["return_loss"] = True
        
        return processed_inputs


def setup_logging(training_args: TrainingArguments) -> None:
    """
    设置日志配置
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def load_dataset(data_args: DataTrainingArguments) -> "Dataset":
    """
    加载数据集
    """
    from datasets import load_from_disk
    import numpy as np
    
    logger.info(f"正在加载数据集: {data_args.dataset_path}")
    dataset = load_from_disk(data_args.dataset_path)
    
    # 过滤掉所有 mask 均为 None 的样本
    def filter_none_masks(samples):
        """过滤掉所有 mask 均为 None 的样本"""
        # 检查 grid_mask 中是否存在至少一个非 None 的 mask
        value_list = []
        for sample in samples:
            # value_list.append(any(mask is not None for mask in sample['obj2mask'].values()))
            print(sample)
            for mask in sample['obj2mask'].values():
                if mask is not None:
                    value_list.append(True)
                    break
            else:
                value_list.append(False)
        return value_list
    
    # logger.info(f"过滤前数据集大小: {len(dataset)}")
    # dataset = dataset.filter(filter_none_masks)
    # logger.info(f"过滤后数据集大小: {len(dataset)}")
    return dataset


def load_model_and_processor(
    model_args: ModelArguments,
    data_args: DataTrainingArguments
) -> tuple["CLIPModel", CLIPProcessor]:
    """
    加载模型和处理器
    """
    from transformers import CLIPModel, CLIPProcessor
    
    logger.info(f"正在加载模型: {model_args.model_name_or_path}")
    
    # 加载模型
    model = CLIPModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # 加载处理器
    processor = CLIPProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # 冻结模型参数（如果需要）
    if model_args.freeze_vision_model:
        logger.info("正在冻结视觉模型参数")
        for param in model.vision_model.parameters():
            param.requires_grad = False
    
    if model_args.freeze_text_model:
        logger.info("正在冻结文本模型参数")
        for param in model.text_model.parameters():
            param.requires_grad = False
    
    return model, processor

def train(
    model: "CLIPModel",
    processor: CLIPProcessor,
    dataset: "Dataset",
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> None:
    """
    执行训练
    """
    from transformers import Trainer
    
    # 自定义Trainer类，用于实现自定义损失函数
    class CustomCLIPTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            自定义损失计算
            您可以在这里实现自己的损失函数逻辑
            """
            # 修改inputs，确保模型返回vision_model_output
            inputs_copy = inputs.copy()
            
            # 移除模型forward方法不需要的参数
            fine_grained_params = ['obj_input_list', 'obj_attention_mask', 'obj_counts', 'grid_mask']
            for param in fine_grained_params:
                if param in inputs_copy:
                    inputs_copy.pop(param)
            
            inputs_copy['output_attentions'] = True
            inputs_copy['output_hidden_states'] = True
            # 获取模型输出
            outputs = model(**inputs_copy)
            # origin instance loss
            instance_loss = outputs.loss
            
            # 细粒度mask loss (参考clip_hal_raw_feature.py的实现)
            obj_loss_sum = 0.0
             # 获取对象级别的文本输入
            obj_input_list = inputs['obj_input_list']
            
            # 获取网格掩码（如果有）
            grid_mask = inputs["grid_mask"]
            for i in range(len(grid_mask)):
                mask = grid_mask[i]
                # Convert list of 24x24 numpy arrays to tensor of shape (24*24, N)
                # where N is the number of masks
                # Stack the mask arrays and reshape to (24*24, N)
                mask_stacked = np.stack(mask, axis=1)  # Shape: (24, 24, N)
                mask_tensor = torch.from_numpy(mask_stacked).float()  # Convert to tensor
                sequence_mask = mask_tensor.view(24*24, -1)  # Reshape to (24*24, N)
                # 计算对象级别的文本嵌入
                # 需要遍历所有的instance
                obj_outputs = model.text_model(
                    input_ids=obj_input_list[i]['input_ids'],
                    attention_mask=obj_input_list[i]['attention_mask']
                )
                obj_text_embeds = obj_outputs.pooler_output
                obj_text_embeds = model.text_projection(obj_text_embeds)
                obj_text_embeds = F.normalize(obj_text_embeds, dim=-1)
                image_embeds = outputs.vision_model_output.hidden_states[-1][i, 1:, :]
                image_embeds = model.visual_projection(image_embeds)
                image_embeds = F.normalize(image_embeds, dim=-1)
                obj_logits_per_text = torch.matmul(obj_text_embeds, image_embeds.t().to(obj_text_embeds.device))
                obj_logits_per_text = obj_logits_per_text * model.logit_scale.exp().to(obj_text_embeds.device)
                logits_per_patch = obj_logits_per_text.t()
                fg_loss = logits_per_patch * sequence_mask.to(logits_per_patch.device)
                obj_loss_sum += fg_loss.mean()
            avg_obj_loss = -obj_loss_sum / len(grid_mask)

            
            # 合并实例级损失和细粒度损失
            alpha = 0.5  # 细粒度损失的权重
            loss = instance_loss + alpha * avg_obj_loss
            
            return loss
    
    # 创建数据整理器
    collator = CLIPCollator(processor, data_args)
    
    # 创建Trainer
    trainer = CustomCLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )
    
    # 检查是否有之前的检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"检测到检查点: {last_checkpoint}")
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # 保存模型
    logger.info("训练完成，保存模型...")
    trainer.save_model()
    trainer.save_state()
    
    # 评估模型
    if training_args.do_eval and "validation" in dataset:
        logger.info("开始评估...")
        metrics = trainer.evaluate()
        logger.info(f"评估结果: {metrics}")


def main() -> None:
    """
    主函数
    """
    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从JSON文件加载参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 从命令行加载参数
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # 2. 设置种子
    set_seed(training_args.seed)
    
    # 3. 设置日志
    setup_logging(training_args)
    
    # 4. 加载数据集
    dataset = load_dataset(data_args)
    
    # 5. 加载模型和处理器
    model, processor = load_model_and_processor(model_args, data_args)
    
    # 6. 执行训练
    train(model, processor, dataset, model_args, data_args, training_args)
    
    logger.info("CLIP微调完成!")


if __name__ == "__main__":
    main()
