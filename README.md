# 准备步骤
**1. 调整配置文件**
按照Repo中的相对位置，将需要修改的配置放入mmdetection文件夹中。
接着，调整目录
```python
cd mmdetection
```

**2. 下载模型权重**
- Faster-R-CNN预训练权重：
```python
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/
```
- 本地服务器训练以后的Faster-R-CNN模型权重文件
  - dropbox下载
```python
wget https://www.dropbox.com/scl/fi/sgtkp0by8p7f4u4c0fh0i/epoch_8.pth?rlkey=zmvpwjce9anpqx6o03gvubvi3&st=s6q8s1c9&dl=1
```
  - 或者，Repo中下载：weights/faster-r-cnn/下的文件，全部下载，再解压epoch_8.zip

**3. 制作数据集**
YOLO需要COCO类型数据，将VOC数据集转换为COCO数据集。


# Faster-R-CNN
**1. 训练**
使用Faster-R-CNN预训练权重进行训练。
```python
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py
```

**2. 测试**
```python
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py epoch_8.pth --out results.pkl
```

# YOLO V3
**1. 训练**
```python
python tools/train.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py
```

**2. 测试**
```python
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py epoch_8.pth --out results.pkl
```
