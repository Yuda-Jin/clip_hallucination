from PIL import Image
import requests
from datasets import load_from_disk
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPModel

# 预定义的颜色列表，确保每次绘图颜色配置一致
GLOBAL_CATEGORY_COLORS = np.array([
    [0.2980392156862745, 0.4470588235294118, 0.6901960784313725],  # 蓝色
    [0.3333333333333333, 0.6588235294117647, 0.40784313725490196], # 绿色
    [0.7686274509803922, 0.3058823529411765, 0.3215686274509804],  # 红色
    [0.5058823529411764, 0.4470588235294118, 0.7019607843137254],  # 紫色
    [0.8, 0.7254901960784313, 0.4549019607843137],                 # 金色
    [0.39215686274509803, 0.7098039215686275, 0.803921568627451],  # 青色
    [0.8666666666666667, 0.5176470588235295, 0.3215686274509804],  # 橙色
    [0.5568627450980392, 0.8235294117647058, 0.7803921568627451],  # 薄荷绿
    [0.5450980392156862, 0.0, 0.13333333333333333],               # 酒红色
    [0.0, 0.5019607843137255, 0.5019607843137255],                # 蓝绿色
    [0.6509803921568628, 0.33725490196078434, 0.7058823529411765], # 粉色
    [0.9019607843137255, 0.6705882352941176, 0.0],                # 琥珀色
    [0.9450980392156862, 0.2980392156862745, 0.23529411764705882]  # 深粉色
])

def plot_logits_per_token(logits_tensor: torch.Tensor, save_path: str = 'results/logits_per_token_lineplot.png', 
                         show_plot: bool = True, figsize: tuple = (12, 8), dpi: int = 300) -> None:
    """
    绘制logits_per_token的折线图
    
    Args:
        logits_tensor: 形状为 [N, 13] 的logits张量
        save_path: 保存图表的路径
        show_plot: 是否显示图表
        figsize: 图表尺寸
        dpi: 保存图表的分辨率
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 将logits_per_token转换为numpy数组用于绘图
    logits_array = logits_tensor.detach().cpu().numpy()
    
    # 创建折线图
    plt.figure(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, min(13, 20)))
    
    # 绘制13条不同颜色的折线
    for i in range(min(13, logits_array.shape[1])):
        plt.plot(logits_array[:, i], label=f'类别 {i+1}', color=colors[i], linewidth=1.5)
    
    # 设置图表属性
    plt.title('logits_per_token 折线图')
    plt.xlabel('Token索引')
    plt.ylabel('Logits值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"折线图已保存到 {save_path}")
    
    # 显示图表
    if show_plot:
        plt.show()

def plot_raw_feature_tsne(vision_features: torch.Tensor, text_features: torch.Tensor, 
                         logits_tensor: torch.Tensor, gt_count: int, 
                         save_path: str = 'results/feature_tsne_visualization.png', 
                         show_plot: bool = True, figsize: tuple = (14, 10), 
                         dpi: int = 300, perplexity: int = 15) -> None:
    """
    使用t-SNE算法直接对特征值进行可视化，包含图像特征和文本特征
    
    Args:
        vision_features: 形状为 [N, feature_dim] 的视觉特征张量，N=577
        text_features: 形状为 [13, feature_dim] 的文本特征张量
        logits_tensor: 形状为 [N, 13] 的logits张量，用于确定相似度最大的类别
        gt_count: ground truth对象的数量，用于区分真实对象和幻觉对象
        save_path: 保存图表的路径
        show_plot: 是否显示图表
        figsize: 图表尺寸
        dpi: 保存图表的分辨率
        perplexity: t-SNE的困惑度参数
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 将张量转换为numpy数组
    vision_array = vision_features.detach().cpu().numpy()
    text_array = text_features.detach().cpu().numpy()
    logits_array = logits_tensor.detach().cpu().numpy()
    
    # 找出每个图像token相似度最大的文本类别
    max_similarity_indices = np.argmax(logits_array, axis=1)
    
    # 使用预定义的全局颜色列表
    num_categories = logits_array.shape[1]
    # 确保颜色列表足够长，如果类别数量超过预定义颜色数量，则循环使用
    category_colors = GLOBAL_CATEGORY_COLORS[:num_categories]
    
    # 为每个图像token点分配对应的颜色
    point_colors = category_colors[max_similarity_indices]
    
    # 合并视觉特征和文本特征进行t-SNE降维
    combined_features = np.vstack((vision_array, text_array))
    print(f"开始t-SNE降维，处理 {combined_features.shape[0]} 个特征向量点（{vision_array.shape[0]}个图像特征 + {text_array.shape[0]}个文本特征）...")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, 
                random_state=42, init='random', metric='euclidean')
    embedded_data = tsne.fit_transform(combined_features)
    
    # 分离图像特征点和文本特征点
    vision_embedded = embedded_data[:vision_array.shape[0]]
    text_embedded = embedded_data[vision_array.shape[0]:]
    
    # 分离第一个图像特征点（全局特征）和其他图像特征点
    first_vision_point = vision_embedded[0:1]  # 第一个点是全局特征
    other_vision_points = vision_embedded[1:]  # 其他576个图像特征点
    other_point_colors = point_colors[1:]      # 对应的颜色
    
    # 创建可视化
    plt.figure(figsize=figsize)
    
    # 根据相似度最大的类别绘制图像token点
    for i in range(num_categories):
        # 找到属于当前类别的点（排除第一个全局特征点）
        mask = (max_similarity_indices[1:] == i)
        if np.any(mask):
            # 计算当前类别的点数量
            count = np.sum(mask)
            # 对于幻觉类别的token，使用三角形标记
            is_halucination = i >= gt_count
            marker = '^' if is_halucination else 'o'
            plt.scatter(other_vision_points[mask, 0], 
                       other_vision_points[mask, 1],
                       c=[category_colors[i]], alpha=0.7, s=60, 
                       marker=marker,
                       label=f'图像特征 - 类别 {i+1} (数量: {count}{" - 幻觉" if is_halucination else ""})')
    
    # 高亮显示第一个图像特征点（全局特征）用紫色
    plt.scatter(first_vision_point[:, 0], first_vision_point[:, 1], 
               c='purple', alpha=1.0, s=200, edgecolors='black', linewidths=2, 
               label='全局特征点')
    
    # 绘制文本特征点，使用对应的类别颜色，用方框标记，无边框，大小与图像特征点一致
    for i in range(num_categories):
        is_halucination = i >= gt_count
        plt.scatter(text_embedded[i, 0], text_embedded[i, 1],
                   c=[category_colors[i]], alpha=0.9, s=60, marker='s',
                   label=f'文本特征 - 类别 {i+1} ({"幻觉" if is_halucination else "真实"})')
    
    # 设置图表属性
    plt.title('特征值的 t-SNE 可视化（包含图像特征和文本特征）')
    plt.xlabel('t-SNE 维度 1')
    plt.ylabel('t-SNE 维度 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"t-SNE 可视化图已保存到 {save_path}")
    
    # 显示统计信息
    print("各文本类别的图像token点分布:")
    hallucination_total = 0
    for i in range(num_categories):
        count = np.sum(max_similarity_indices == i)
        category_type = "真实" if i < gt_count else "幻觉"
        print(f"类别 {i+1} ({category_type}): {count} 个点")
        if category_type == "幻觉" and count > 0:
            hallucination_total += count
    # 打印幻觉token的总数量
    print(f"\n幻觉token的总数量: {hallucination_total} 个")
    
    # 显示图表
    if show_plot:
        plt.show()

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
    """
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336",local_files_only=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336",local_files_only=True)
ds = load_from_disk('E:\DATA\dataset\OHD-Caps-train')
sample = ds[1]

image = sample['image']
# 为每个对象添加prompt模板
prompt_template = "There is {}."
total_obj = [prompt_template.format(obj) for obj in sample['ground_truth'] + sample['hal_obj']]
print('gt_num:',len(sample['ground_truth']))
print(total_obj)

inputs = processor(text=total_obj, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
vision_output = outputs.vision_model_output.last_hidden_state#[:,1:,:]
projed_vision = model.visual_projection(vision_output)
projed_vision = projed_vision / _get_vector_norm(projed_vision)
projed_vision = projed_vision.squeeze(0)
projed_text = outputs.text_embeds
logits_per_text = torch.matmul(projed_text, projed_vision.t().to(projed_vision.device))
logits_per_text = logits_per_text * model.logit_scale.exp().to(projed_text.device)

logits_per_token = logits_per_text.t()#[:20,:]

# text_self_score = torch.matmul(projed_text, projed_text.t().to(projed_text.device))
# text_self_score = text_self_score * model.logit_scale.exp().to(projed_text.device)

# plot_logits_per_token(logits_per_token)

# 绘制t-SNE可视化，传入真实对象的数量
gt_count = len(sample['ground_truth'])
plot_raw_feature_tsne(projed_vision, projed_text, logits_per_token, gt_count)

a=1
# global_logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# global_probs = logits_per_image#.softmax(dim=1) # we can take the softmax to get the label probabilities
# print(global_logits_per_image)
