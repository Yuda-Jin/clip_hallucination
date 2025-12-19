import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model(    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path='/root/autodl-tmp/cache/modelscope/models/facebook/sam3/sam3.pt',
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False)
processor = Sam3Processor(model)
# Load an image
image = Image.open("assets/COCO_train2014_000000399138.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="woman")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(masks.shape)
print(masks)

# 处理掩码：合并所有对象的掩码
import numpy as np
import matplotlib.pyplot as plt

# 将张量移动到CPU并转换为numpy数组
try:
    masks_np = masks.cpu().numpy()
    
    # 移除额外的通道维度 (O*1*H*W -> O*H*W)
    masks_np = np.squeeze(masks_np, axis=1)
    
    # 合并所有掩码：取最大值（如果有重叠，保留所有对象）
    # 如果只想保留一个对象，可以使用np.max
    merged_mask = np.max(masks_np, axis=0)
    
    # 确保形状为H*W*1
    merged_mask = merged_mask[..., np.newaxis]
    
    print(f"合并后的掩码形状: {merged_mask.shape}")
    
    # 绘制掩码
    plt.figure(figsize=(10, 10))
    plt.imshow(np.squeeze(merged_mask), cmap='gray')  # 显示时移除通道维度
    plt.title('Merged Object Masks')
    plt.axis('off')
    
    # 保存掩码图像
    plt.savefig('merged_mask.png', bbox_inches='tight', dpi=100)
    print("掩码图像已保存为: merged_mask.png")
    
    # 显示图像
    plt.show()
    
    # 也可以保存为numpy数组
    # np.save('merged_mask.npy', merged_mask)
    # print("掩码数组已保存为: merged_mask.npy")
    
except Exception as e:
    print(f"处理掩码时出错: {e}")
    import traceback
    traceback.print_exc()

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]
# Results contain:
# - masks: Binary masks resized to original image size
# - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
# - scores: Confidence scores

# #模型下载
# from modelscope import snapshot_download
# import os
# # 设置自定义缓存路径
# os.environ['MODELSCOPE_CACHE'] = 'E:\DATA\model'
# model_dir = snapshot_download('facebook/sam3')