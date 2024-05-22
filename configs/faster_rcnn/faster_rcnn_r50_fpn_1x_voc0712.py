_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn voc.py',
    '../_base_/datasets/voc0712 copy1.py',
    '../_base_/schedules/schedule_1x copy1.py', '../_base_/default_runtime copy1.py'
]

# 模型设置
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))  # VOC数据集有20个类别

# 预训练权重
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
