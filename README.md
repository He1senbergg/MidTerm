# 调整配置文件
按照Repo中的相对位置，将需要修改的配置放入mmdetection文件夹中。
接着，调整目录
```python
cd mmdetection
```

# Faster-R-CNN
**1. 下载预训练权重模型**
```python
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/
```

**2. 训练**
```python
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py
```

**3. 测试**
```python
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py epoch_8.pth --out results.pkl
```

# YOLO V3
**1. 制作数据集**
YOLO需要COCO类型数据，将VOC数据集转换为COCO数据集。

**2. 训练**
```python
python tools/train.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py
```

**3. 测试**
```python
python tools/test.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py epoch_273.pth --out results.pkl
```
