#!/usr/bin/env python
# coding=utf-8
"""
CLIP微调脚本（支持细粒度对象-区域对齐损失，并行化实现）
"""

from pathlib import Path
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor
import functools

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
    CLIPProcessor,
    CLIPModel
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from datasets import Dataset

# 检查Transformers版本
check_min_version("4.26.0")
require_version("datasets>=1.8.0")

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
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
    max_objects: int = field(
        default=20, metadata={"help": "每个样本最多对象数（用于padding）"}
    )


class CLIPCollator:
    def __init__(self, processor: CLIPProcessor, data_args: DataTrainingArguments):
        self.processor = processor
        self.data_args = data_args
        self.max_objects = data_args.max_objects
        # 存储线程池配置，延迟初始化线程池以支持序列化
        self.max_workers = data_args.preprocessing_num_workers or os.cpu_count()
        self._executor = None

    @property
    def executor(self):
        """延迟初始化线程池，确保训练期间始终保持活跃"""
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self._executor

    def __del__(self):
        """清理线程池资源"""
        if hasattr(self, '_executor') and self._executor is not None:
            try:
                self._executor.shutdown(wait=True)
            except:
                pass

    def __getstate__(self):
        """自定义序列化逻辑，排除无法被pickle的线程池对象"""
        state = self.__dict__.copy()
        del state['_executor']
        return state

    def __setstate__(self, state):
        """自定义反序列化逻辑，重建线程池"""
        self.__dict__.update(state)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def process_example(self, example):
        """处理单个示例，提取对象提示和网格掩码"""
        obj_prompts = []
        grid_masks = []
        for obj, mask in example.get("obj2grid", {}).items():
            if mask is None:
                continue
            obj_prompts.append(f"There is {obj}.")
            grid_masks.append(mask)
        return obj_prompts, grid_masks

    def __call__(self, examples: List[Dict]) -> Dict:
        start_time = time.time()
        
        images = [example["image"] for example in examples]
        captions = [str(example.get("positive_sample", "")) for example in examples]

        # 并行处理所有示例
        results = list(self.executor.map(self.process_example, examples))
        
        all_obj_prompts = [r[0] for r in results]
        all_grid_masks = [r[1] for r in results]

        max_obj_num = max(len(prompts) for prompts in all_obj_prompts) if all_obj_prompts else 0
        obj_valid_masks = []
        
        # 预分配零掩码
        zero_mask = np.zeros((24, 24), dtype=np.float32)
        
        # 填充对象提示和网格掩码
        for obj_prompts, grid_masks in zip(all_obj_prompts, all_grid_masks):
            real_num = len(obj_prompts)
            valid_mask = [1] * real_num + [0] * (max_obj_num - real_num)
            obj_valid_masks.append(valid_mask)

            # 高效填充
            num_to_pad = max_obj_num - len(obj_prompts)
            if num_to_pad > 0:
                obj_prompts.extend([""] * num_to_pad)
                grid_masks.extend([zero_mask] * num_to_pad)
            
            # 堆叠网格掩码
            grid_masks[:] = np.stack(grid_masks, axis=0)

        # 分块处理tokenization以减少内存占用
        flat_prompts = [p for prompts in all_obj_prompts for p in prompts]
        chunk_size = 1000  # 可根据内存调整
        all_input_ids = []
        all_attention_masks = []
        
        for i in range(0, len(flat_prompts), chunk_size):
            chunk_prompts = flat_prompts[i:i+chunk_size]
            chunk_tok = self.processor.tokenizer(
                chunk_prompts,
                padding="max_length",
                truncation=True,
                max_length=self.data_args.max_seq_length,
                return_tensors="pt"
            )
            all_input_ids.append(chunk_tok["input_ids"])
            all_attention_masks.append(chunk_tok["attention_mask"])
        
        # 拼接结果
        full_input_ids = torch.cat(all_input_ids, dim=0) if all_input_ids else torch.empty(0, self.data_args.max_seq_length, dtype=torch.long)
        full_attention_mask = torch.cat(all_attention_masks, dim=0) if all_attention_masks else torch.empty(0, self.data_args.max_seq_length, dtype=torch.long)

        B = len(examples)
        obj_input_ids = full_input_ids.view(B, max_obj_num, -1)
        obj_attention_mask = full_attention_mask.view(B, max_obj_num, -1)

        # 处理主标题和图像
        text_inputs = self.processor.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.data_args.max_seq_length,
            return_tensors="pt"
        )
        image_inputs = self.processor.image_processor(images, return_tensors="pt")

        elapsed_time = time.time() - start_time
        logger.info(f"CLIPCollator call completed in {elapsed_time:.4f} seconds for {len(examples)} examples")
        
        return {
            **text_inputs,
            **image_inputs,
            "obj_input_ids": obj_input_ids,
            "obj_attention_mask": obj_attention_mask,
            "grid_mask": torch.from_numpy(np.stack(all_grid_masks)).float(),
            "obj_valid_mask": torch.tensor(obj_valid_masks, dtype=torch.bool),
            "return_loss": True
        }


def setup_logging(training_args: TrainingArguments) -> None:
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


def load_dataset(data_args: DataTrainingArguments) -> Dataset:
    from datasets import load_from_disk
    logger.info(f"正在加载数据集: {data_args.dataset_path}")
    dataset = load_from_disk(data_args.dataset_path)
    return dataset


def load_model_and_processor(
    model_args: ModelArguments,
    data_args: DataTrainingArguments
) -> tuple[CLIPModel, CLIPProcessor]:
    logger.info(f"正在加载模型: {model_args.model_name_or_path}")
    model = CLIPModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    processor = CLIPProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

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
    model: CLIPModel,
    processor: CLIPProcessor,
    dataset: Dataset,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> None:
    class CustomCLIPTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # --- 原始 CLIP 损失 ---
            base_keys = {"input_ids", "attention_mask", "pixel_values", "return_loss"}
            base_inputs = {k: v for k, v in inputs.items() if k in base_keys}
            base_inputs["output_hidden_states"] = True
            outputs = model(**base_inputs)
            instance_loss = outputs.loss

            # --- 细粒度损失（向量化）---
            B = inputs["input_ids"].size(0)
            N = inputs["obj_input_ids"].size(1)
            num_patches = outputs.vision_model_output.hidden_states[-1].size(1) - 1  # exclude cls token
            patch_grid_h = patch_grid_w = int(num_patches ** 0.5)

            # 文本嵌入 (B, N, D)
            flat_input_ids = inputs["obj_input_ids"].view(B * N, -1)
            flat_attn_mask = inputs["obj_attention_mask"].view(B * N, -1)
            text_out = model.text_model(input_ids=flat_input_ids, attention_mask=flat_attn_mask)
            text_embeds = model.text_projection(text_out.pooler_output)
            text_embeds = F.normalize(text_embeds, dim=-1).view(B, N, -1)

            # 图像 patch 嵌入 (B, P, D)
            img_patch_embeds = outputs.vision_model_output.hidden_states[-1][:, 1:, :]
            img_patch_embeds = model.visual_projection(img_patch_embeds)
            img_patch_embeds = F.normalize(img_patch_embeds, dim=-1)

            # 相似度 (B, N, P)
            logits = torch.einsum('bnd,bpd->bnp', text_embeds, img_patch_embeds)
            logits = logits * model.logit_scale.exp()

            # 对齐 mask: (B, N, 24, 24) → (B, N, P)
            grid_mask = inputs["grid_mask"]
            resized_mask = F.adaptive_avg_pool2d(
                grid_mask.view(B * N, 1, 24, 24),
                (patch_grid_h, patch_grid_w)
            ).view(B, N, num_patches)

            # 应用 valid mask
            valid_mask = inputs["obj_valid_mask"].unsqueeze(-1)  # (B, N, 1)
            weighted_logits = logits * resized_mask
            fg_loss_per_obj = weighted_logits.mean(dim=-1)  # (B, N)
            total_valid = valid_mask.sum().clamp(min=1)
            fg_loss = (fg_loss_per_obj * valid_mask.squeeze(-1)).sum() / total_valid

            # 总损失
            alpha = 0.5
            total_loss = instance_loss + alpha * (-fg_loss)
            return total_loss

    collator = CLIPCollator(processor, data_args)
    trainer = CustomCLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"检测到检查点: {last_checkpoint}")

    logger.info("开始训练...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    logger.info("训练完成，保存模型...")
    trainer.save_model()
    trainer.save_state()


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    setup_logging(training_args)
    dataset = load_dataset(data_args)
    model, processor = load_model_and_processor(model_args, data_args)
    train(model, processor, dataset, model_args, data_args, training_args)
    logger.info("CLIP微调完成!")


if __name__ == "__main__":
    main()