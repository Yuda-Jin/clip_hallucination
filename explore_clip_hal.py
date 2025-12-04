from PIL import Image
import requests
from datasets import load_from_disk

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
ds = load_from_disk('/root/autodl-tmp/dataset/OHD-Caps-train')
sample = ds[0]

image = sample['image']
total_obj = sample['ground_truth']+sample['hal_obj']
print('gt_num:',len(sample['ground_truth']))
print(total_obj)

inputs = processor(text=total_obj, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image#.softmax(dim=1) # we can take the softmax to get the label probabilities
print(probs)
