_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

load_from = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
