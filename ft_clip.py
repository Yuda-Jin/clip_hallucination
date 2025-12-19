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
        objs = []
        obj_counts = []  # 记录每个图像对应的对象数量
        grid_mask = []  # 记录每个图像对应的网格掩码
        
        for example in examples:
            current_objs = example["obj"]
            obj_counts.append(len(current_objs))
            objs.extend([f"There is {obj}" for obj in current_objs])
            
            # 获取当前图像的网格掩码
            # 注意：这里需要根据实际情况调整grid_mask的格式
            # 目前grid_mask存储的是obj2grid.values()，即每个对象对应的网格
            # 后续可能需要将其转换为适合模型使用的格式
            current_grid_mask = example["obj2grid"].values()
            grid_mask.append(current_grid_mask)
        
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
        
        obj_inputs = self.processor.tokenizer(
            objs,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_seq_length,
            return_tensors="pt"
        )
        
        # 处理图像
        image_inputs = self.processor.image_processor(
            images,
            return_tensors="pt"
        )
        
        # 合并所有输入
        processed_inputs = {
            **text_inputs,
            **image_inputs,
            "obj_input_ids": obj_inputs["input_ids"],
            "obj_attention_mask": obj_inputs["attention_mask"],
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
    
    # 定义map函数：如果label存在grid-mask则使用grid-mask，否则分配全为0的mask
    def process_mask(example):
        # 检查example中是否有'label'字段
        obj2grid = {}
        for obj in example["mask"]:
            obj2grid[obj["label"]]=obj["grid"]
        for obj in example["hal_obj"]:
            obj2grid[obj]=np.zeros((24, 24), dtype=int)
        example["obj2grid"] = obj2grid
        example["obj"] = list(obj2grid.keys())  # 将dict_keys转换为列表
        return example
    
    logger.info(f"正在处理数据集的grid...")
    dataset = dataset.map(
        function=process_mask,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        desc=f"Processing grids for dataset",
        load_from_cache_file=not data_args.overwrite_cache
    )

    logger.info(f"数据集加载完成: {dataset}")
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
            fine_grained_params = ['obj_input_ids', 'obj_attention_mask', 'obj_counts', 'grid_mask']
            for param in fine_grained_params:
                if param in inputs_copy:
                    inputs_copy.pop(param)
            
            inputs_copy['output_attentions'] = True
            inputs_copy['output_hidden_states'] = True
            
            # 获取模型输出
            outputs = model(**inputs_copy)
            
            # origin instance loss
            # 如果模型已经计算了损失，直接使用
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                instance_loss = outputs.loss
            else:
                # 自定义损失计算逻辑示例
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # 标准CLIP损失（图像到文本和文本到图像的交叉熵损失）
                image_loss = torch.nn.functional.cross_entropy(
                    logits_per_image, 
                    torch.arange(len(logits_per_image), device=logits_per_image.device)
                )
                text_loss = torch.nn.functional.cross_entropy(
                    logits_per_text, 
                    torch.arange(len(logits_per_text), device=logits_per_text.device)
                )
                
                # 合并损失
                instance_loss = (image_loss + text_loss) / 2
            
            # 细粒度mask loss (参考clip_hal_raw_feature.py的实现)
            fine_grained_loss = 0.0
            if hasattr(outputs, 'vision_model_output') and 'obj_input_ids' in inputs:
                try:
                    # 获取对象级别的文本输入
                    obj_input_ids = inputs['obj_input_ids']
                    obj_attention_mask = inputs['obj_attention_mask']
                    
                    # 获取网格掩码（如果有）
                    grid_mask = inputs.get('grid_mask', None)
                    
                    # 计算对象级别的文本嵌入
                    obj_outputs = model.text_model(
                        input_ids=obj_input_ids,
                        attention_mask=obj_attention_mask
                    )
                    obj_text_embeds = obj_outputs.last_hidden_state[:, 0, :]  # 获取[CLS]标记的嵌入
                    obj_text_embeds = model.text_projection(obj_text_embeds)
                    obj_text_embeds = F.normalize(obj_text_embeds, dim=-1)
                
                    # 获取视觉特征图 - 去掉CLS标记，得到网格特征 [b, p*p, n]
                    vision_output = outputs.vision_model_output.last_hidden_state  # (batch_size, num_patches+1, hidden_size)
                    vision_output = vision_output[:, 1:, :]  # 去掉CLS标记 (batch_size, p*p, hidden_size)
                    
                    # 将视觉特征图通过视觉投影层得到投影后的视觉特征
                    projed_vision = model.visual_projection(vision_output)  # (batch_size, p*p, projection_dim)
                    
                    # 对投影后的视觉特征进行归一化
                    projed_vision = F.normalize(projed_vision, dim=-1)
                    
                    # 细粒度对比损失计算
                    # 由于每张图片的obj数量不一致，暂时不进行批量计算，逐张处理
                    batch_size = projed_vision.size(0)
                    total_fine_grained_loss = 0.0
                    loss_count = 0
                    
                    # 假设inputs中包含了每个图像对应的对象数量信息
                    # 如果没有，需要从CLIPCollator中传递这个信息
                    # 这里先假设每个图像的对象数量存储在inputs['obj_counts']中
                    if 'obj_counts' in inputs:
                        obj_counts = inputs['obj_counts']
                        obj_start_idx = 0
                        
                        for i in range(batch_size):
                            # 获取当前图像的视觉特征 [p*p, n]
                            current_vision = projed_vision[i]  # (p*p, n)
                            
                            # 获取当前图像对应的对象数量
                            num_objs = obj_counts[i]
                            if num_objs == 0:
                                continue
                            
                            # 获取当前图像对应的对象文本嵌入 [num_objs, n]
                            current_obj_text = obj_text_embeds[obj_start_idx:obj_start_idx+num_objs]  # (num_objs, n)
                            obj_start_idx += num_objs
                            
                            # 使用grid_mask挑选对应网格（如果提供了）
                            if grid_mask is not None and grid_mask[i] is not None:
                                # 这里需要根据grid_mask的具体格式来挑选网格
                                # 假设grid_mask[i]是一个布尔掩码，形状为[p*p]
                                selected_vision = current_vision[grid_mask[i]]  # (selected_patches, n)
                            else:
                                # 如果没有grid_mask，使用所有网格
                                selected_vision = current_vision  # (p*p, n)
                            
                            # 计算当前图像的视觉特征和对象文本的相似度
                            # 相似度矩阵形状: [selected_patches, num_objs]
                            similarity = torch.matmul(selected_vision, current_obj_text.t())
                            similarity = similarity * model.logit_scale.exp()
                            
                            # 计算细粒度对比损失
                            # 对于每个网格特征，它应该与所有属于该图像的对象匹配
                            num_visual = selected_vision.size(0)
                            labels = torch.ones(num_visual, num_objs, device=similarity.device)
                            
                            # 使用二元交叉熵损失
                            probs = torch.sigmoid(similarity)
                            loss = F.binary_cross_entropy(probs, labels)
                            
                            total_fine_grained_loss += loss
                            loss_count += 1
                        
                        if loss_count > 0:
                            fine_grained_loss = total_fine_grained_loss / loss_count
                    else:
                        # 简化版：如果没有obj_counts信息，使用原来的计算方式
                        # 但这里会假设每个图像的对象数量相同
                        num_visual_features = projed_vision.size(1)  # p*p = 576
                        num_objects_total = obj_text_embeds.size(0)
                        
                        # 假设每个图像的对象数量相同
                        num_objects_per_image = num_objects_total // batch_size
                        
                        if num_objects_per_image > 0:
                            # 创建标签矩阵
                            labels = torch.zeros(batch_size * num_visual_features, num_objects_total, device=projed_vision.device)
                            
                            for i in range(batch_size):
                                # 当前图像的对象索引范围
                                start_obj = i * num_objects_per_image
                                end_obj = start_obj + num_objects_per_image
                                
                                # 当前图像的视觉特征索引范围
                                start_visual = i * num_visual_features
                                end_visual = start_visual + num_visual_features
                                
                                # 标记当前图像的视觉特征与对象的匹配关系
                                labels[start_visual:end_visual, start_obj:end_obj] = 1.0
                            
                            # 计算视觉特征和对象文本的相似度
                            flattened_vision = projed_vision.view(batch_size * num_visual_features, -1)  # [b*p*p, n]
                            similarity_matrix = torch.matmul(flattened_vision, obj_text_embeds.t())  # [b*p*p, total_objs]
                            similarity_matrix = similarity_matrix * model.logit_scale.exp()
                            
                            # 使用二元交叉熵损失
                            probs = torch.sigmoid(similarity_matrix)
                            fine_grained_loss = F.binary_cross_entropy(probs, labels)
                    
                except Exception as e:
                    # 如果细粒度损失计算失败，跳过这部分损失
                    logger.warning(f"细粒度损失计算失败: {e}")
                    fine_grained_loss = 0.0
            
            # 合并实例级损失和细粒度损失
            alpha = 0.5  # 细粒度损失的权重
            loss = instance_loss + alpha * fine_grained_loss
            
            return (loss, outputs) if return_outputs else loss
    
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
