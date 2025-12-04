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
print(masks)

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