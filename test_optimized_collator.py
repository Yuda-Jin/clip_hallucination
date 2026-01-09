#!/usr/bin/env python
# coding=utf-8
"""
Simple test script to verify the optimized CLIPCollator functionality
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from ft_clip import CLIPCollator, DataTrainingArguments
from PIL import Image
import tempfile
import os

def test_collator():
    print("Testing the optimized CLIPCollator...")
    
    # Load a pre-trained CLIP model and processor for testing
    print("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create mock data training arguments
    data_args = DataTrainingArguments(
        dataset_path="",  # Not needed for this test
        image_column="image",
        caption_column="caption",
        max_seq_length=77,
        preprocessing_num_workers=4  # Test with multiple workers
    )
    
    # Create the optimized collator
    collator = CLIPCollator(processor, data_args)
    
    # Create mock examples for testing
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    # Create example data with object regions
    examples = [
        {
            "image": dummy_image,
            "positive_sample": "A red square",
            "obj2grid": {
                "square": np.ones((24, 24), dtype=np.float32)  # Simulate a mask
            }
        },
        {
            "image": dummy_image,
            "positive_sample": "Another red square",
            "obj2grid": {
                "square": np.ones((24, 24), dtype=np.float32) * 0.5,
                "background": np.ones((24, 24), dtype=np.float32) * 0.3
            }
        }
    ]
    
    print("Processing examples with optimized collator...")
    result = collator(examples)
    
    print(f"Result keys: {list(result.keys())}")
    print(f"Input IDs shape: {result['input_ids'].shape}")
    print(f"Pixel values shape: {result['pixel_values'].shape}")
    print(f"Object input IDs shape: {result['obj_input_ids'].shape}")
    print(f"Grid mask shape: {result['grid_mask'].shape}")
    print(f"Valid mask shape: {result['obj_valid_mask'].shape}")
    
    # Verify that the shapes are as expected
    assert result['input_ids'].shape[0] == 2, f"Expected batch size 2, got {result['input_ids'].shape[0]}"
    assert result['obj_input_ids'].shape[0] == 2, f"Expected batch size 2, got {result['obj_input_ids'].shape[0]}"
    assert result['grid_mask'].shape[0] == 2, f"Expected batch size 2, got {result['grid_mask'].shape[0]}"
    
    print("All tests passed! The optimized collator is working correctly.")
    
    # Clean up
    del collator

if __name__ == "__main__":
    test_collator()