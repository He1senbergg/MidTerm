# Faster-R-CNN
**1. 调整配置文件**
按照Repo中的相对位置，放入mmdetection文件夹中。
接着，调整目录
```python
cd mmdetection
```

**2. 下载预训练权重模型**
```python
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/
```

**3. 训练**
```python
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py
```

**4. 测试**
```python
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py work_dirs/faster_rcnn_r50_fpn_1x_voc0712/epoch_8.pth --out results.pkl
```
