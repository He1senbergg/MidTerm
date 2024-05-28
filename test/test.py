import os
import cv2
import mmcv
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mmcv.visualization.image import imshow_det_bboxes
from mmdet.apis import init_detector, inference_detector

def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference with Faster R-CNN and YOLO models and generate comparison images.")
    parser.add_argument('--compare', type=bool, default=False, help='Generate comparison images for Faster R-CNN and YOLO.')
    parser.add_argument('--fasterpth', type=str, default=None, help='Path to the Faster R-CNN model checkpoint.')
    parser.add_argument('--fasterconfig', type=str, default=None, help='Path to the Faster R-CNN config.')
    parser.add_argument('--yolopth', type=str, default=None, help='Path to the YOLO model checkpoint.')
    parser.add_argument('--yoloconfig', type=str, default=None, help='Path to the YOLO config.')
    parser.add_argument('--imgfolder', type=str, default=None, help='Path to the folder containing images to perform inference on.')
    parser.add_argument('--output', type=str, default=None, help='Path to the folder to save the comparison images.')
    parser.add_argument("--bbox_color", type=str, default='white', help="Color of the bounding boxes.")
    parser.add_argument("--text_color", type=str, default='red', help="Color of the class labels.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 是否生成对比图像
    compare = args.compare

    # 配置文件路径
    faster_rcnn_config = args.fasterconfig
    yolov3_config = args.yoloconfig

    # 训练好的模型权重文件路径
    faster_rcnn_checkpoint = args.fasterpth
    yolov3_checkpoint = args.yolopth
    # 检查是否同时提供了模型配置文件和权重文件

    if compare and (faster_rcnn_config is None or yolov3_config is None or \
                    faster_rcnn_checkpoint is None or yolov3_checkpoint is None):
        raise ValueError("--fasterconfig, --fasterpth, --yoloconfig, and --yolopth must be provided together for comparison.")
    if (faster_rcnn_config is None) != (faster_rcnn_checkpoint is None):
        raise ValueError("--fasterconfig and --fasterpth must be provided together.")
    if (yolov3_config is None) != (yolov3_checkpoint is None):
        raise ValueError("--yoloconfig and --yolopth must be provided together.")
    if faster_rcnn_config == None and yolov3_config == None and \
        faster_rcnn_checkpoint == None and yolov3_checkpoint == None:
        raise ValueError("You have to provide at least one model and its config to perform inference.")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 初始化检测模型
    faster_rcnn_model = init_detector(faster_rcnn_config, faster_rcnn_checkpoint, device=device)
    yolov3_model = init_detector(yolov3_config, yolov3_checkpoint, device=device)

    # VOC数据集类别
    class_names = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 图片文件夹路径
    img_folder = args.imgfolder
    if img_folder is None:
        raise ValueError("You have to provide a folder containing images to perform inference on.")
    
    # 结果保存路径
    output_folder = args.output
    if output_folder is None:
        raise ValueError("You have to provide a folder to save the comparison images.")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # 锚框颜色和类别文字颜色
    bbox_color = args.bbox_color
    text_color = args.text_color

    # 对每张图片进行推理并保存对比结果
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)

        # 读取图像
        img = mmcv.imread(img_path)

        # Faster R-CNN 推理
        if faster_rcnn_checkpoint and faster_rcnn_config:
            faster_rcnn_result = inference_detector(faster_rcnn_model, img_path)
            faster_rcnn_bboxes = faster_rcnn_result.pred_instances.bboxes.cpu().numpy()
            faster_rcnn_labels = faster_rcnn_result.pred_instances.labels.cpu().numpy()
            faster_rcnn_scores = faster_rcnn_result.pred_instances.scores.cpu().numpy()
            faster_rcnn_bboxes_with_scores = np.hstack([faster_rcnn_bboxes, faster_rcnn_scores[:, np.newaxis]])

            output_faster = os.path.join(output_folder, f'{img_file[:-4]}_faster_rcnn{img_file[-4:]}')
            # 可视化并保存结果
            faster_rcnn_img = imshow_det_bboxes(
                img.copy(),
                faster_rcnn_bboxes_with_scores,
                faster_rcnn_labels,
                class_names=class_names,
                score_thr=0.3,
                show=False, # 是否显示结果
                bbox_color=bbox_color,  # 锚框颜色设置为白色
                text_color=text_color,  # 类别文字颜色设置为红色
                font_scale=1,  # 设置类别文字的字号缩放率为1
                thickness=3,  # 设置锚框的线条粗细为3
                out_file=output_faster, # 设置输出文件保存位置
            )

        # YOLOv3 推理
        if yolov3_checkpoint and yolov3_config:
            yolov3_result = inference_detector(yolov3_model, img_path)
            yolov3_bboxes = yolov3_result.pred_instances.bboxes.cpu().numpy()
            yolov3_labels = yolov3_result.pred_instances.labels.cpu().numpy()
            yolov3_scores = yolov3_result.pred_instances.scores.cpu().numpy()
            yolov3_bboxes_with_scores = np.hstack([yolov3_bboxes, yolov3_scores[:, np.newaxis]])

            output_yolo = os.path.join(output_folder, f'{img_file[:-4]}_yolov3{img_file[-4:]}')
            yolov3_img = imshow_det_bboxes(
                img.copy(),
                yolov3_bboxes_with_scores,
                yolov3_labels,
                class_names=class_names,
                score_thr=0.3,
                show=False, # 是否显示结果
                bbox_color=bbox_color,  # 锚框颜色设置为白色
                text_color=text_color,  # 类别文字颜色设置为红色
                font_scale=1,  # 设置类别文字的字号缩放率为1
                thickness=3,  # 设置锚框的线条粗细为3
                out_file=output_yolo, # 设置输出文件保存位置
            )

        if compare:
            # 将图像从BGR格式转换为RGB格式
            faster_rcnn_img = cv2.cvtColor(faster_rcnn_img, cv2.COLOR_BGR2RGB)
            yolov3_img = cv2.cvtColor(yolov3_img, cv2.COLOR_BGR2RGB)

            # 创建对比图
            _, axs = plt.subplots(1, 2, figsize=(15, 10), dpi=300)
            axs[0].imshow(faster_rcnn_img)
            axs[0].set_title('Faster R-CNN')
            axs[0].axis('off')
            axs[1].imshow(yolov3_img)
            axs[1].set_title('YOLOv3')
            axs[1].axis('off')

            plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)

            # 保存对比图
            comparison_output_path = os.path.join(output_folder, f"{img_file[:-4]}_Compare{img_file[-4:]}")
            plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()

if __name__ == '__main__':
    main()
