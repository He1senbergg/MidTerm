import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
from mmcv.visualization.image import imshow_det_bboxes

# 配置文件路径
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py'
# 训练好的模型权重文件路径
checkpoint_file = 'epoch_8.pth'

# 初始化检测模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# VOC数据集类别
class_names = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 图片文件夹路径
img_folder = "picture/"
# 结果保存路径
output_folder = "result/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 对每张图片进行推理
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    # 推理
    result = inference_detector(model, img_path)
    # 提取 bbox 和 label 信息
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    
    # 将分数添加到bboxes中
    bboxes_with_scores = np.hstack([bboxes, scores[:, np.newaxis]])

    # 读取图像
    img = mmcv.imread(img_path)
    # 可视化并保存结果
    output_path = os.path.join(output_folder, img_file)
    imshow_det_bboxes(
        img,
        bboxes_with_scores,
        labels,
        class_names=class_names,
        score_thr=0.3,
        out_file=output_path
    )
