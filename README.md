# Repo内容说明
**1. 任务一**

内容为“任务一/”下所示的四个python文件。

**2. 任务二**

其余文件均为任务二相关文件。

# 任务一
## 准备步骤
**1. 代码下载**

下载“任务一/”下的四个python文件，放在同级目录

**2. 可调参数概述**
| 参数名 | 类型 | 默认值 | 描述 |
| ------- | ------- | ------- | ------- |
| `--num_classes` | int | 200 | Number of classes in the dataset. |
| `--data_dir` | str | /mnt/ly/models/deep_learning/mid_term/data/CUB_200_2011 | Path to the CUB-200-2011 dataset directory. |
| `--batch_size` | int | 32 | Batch size for training. |
| `--num_epochs` | int | 70 | Number of epochs for training. |
| `--learning_rate` | float | 0.001 | Learning rate for the optimizer. |
| `--momentum` | float | 0.9 | Momentum for the SGD optimizer. |
| `--weights` | str | IMAGENET1K_V1 | Pytorch中的ResNet预训练权重名称 |
| `--model_path` | str | None | 读取本地pth |
| `--optimizer` | str | SGD | Optimizer to use (SGD or Adam). |
| `--logdir` | str | /mnt/ly/models/deep_learning/mid_term/tensorboard/1 | Directory to save TensorBoard logs. |
| `--save_dir` | str | /mnt/ly/models/deep_learning/mid_term/model | 训练时保存pth的位置 |
| `--scratch` | bool | False | 是否随机初始化 |
| `--decay` | float | 1e-3 | Weight decay for the optimizer. |
| `--milestones` | list | None | List of epochs to decrease the learning rate. |
| `--gamma` | float | 0.1 | 当使用milestones时，每次学习率的缩放率 |

**3. 必须自适应调整的参数**

因为代码中的默认地址信息为服务器上的地址，所以本地运行时，必须在命令行中重新赋值以修改。
- `--data_dir`：改为本地CUB_200_2011的位置（绝对位置）
- `--save_dir`：运行过程中，每当遇到更高的test-accuracy时，model的pth的保存位置（绝对位置）
- `--logdir`：日志的保存地址（绝对位置）

## 2. 训练

命令行运行代码

- 示例1（请注意修改以下的信息的绝对位置）：使用预训练模型与默认参数进行训练
  ```
  python train.py --data_dir /data/CUB_200_2011 --save_dir /model --logdir /tensorboard
  ```
- 示例2（请注意修改以下的信息的绝对位置）：使用预训练模型、Adam与其他默认参数开始训练
  ```
  python train.py --data_dir /data/CUB_200_2011 --save_dir /model --logdir /tensorboard --optimizer Adam
  ```
- 示例3（请注意修改以下的信息的绝对位置）：使用随机初始化与其他默认参数开始训练
  ```
  python train.py --data_dir /data/CUB_200_2011 --save_dir /model --logdir /tensorboard --scratch True
  ```

## 3. 测试

测试时，需要提供两个参数`--data_dir`、`--model_path`。

命令行运行代码，示例如下（请注意修改以下的信息的绝对位置）：
```
python test.py --data_dir /data/CUB_200_2011 --model_path model.pth
```

# 任务二
## 准备步骤
**1. 调整配置文件**

按照Repo中的相对位置，将需要修改的配置放入mmdetection文件夹中。

接着，调整目录
```
cd mmdetection
```

**2. 下载模型权重**

- Faster-R-CNN预训练权重：
```python
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/
```
- 本地服务器训练以后的Faster-R-CNN模型权重文件
  - dropbox下载
  ```
  wget https://www.dropbox.com/scl/fi/sgtkp0by8p7f4u4c0fh0i/epoch_8.pth?rlkey=zmvpwjce9anpqx6o03gvubvi3&st=s6q8s1c9&dl=1
  ```
  - 或者，Repo中下载：weights/faster-r-cnn/下的文件，全部下载，再解压epoch_8.zip

- 本地服务器训练以后的YOLO V3模型权重文件
  - dropbox下载
  ```
  https://www.dropbox.com/scl/fi/qa7ziesle7kxfunojsxcc/epoch_273.pth?rlkey=mopji9d302b0q7gh234v73weg&st=ir8gcntv&dl=1
  ```

**3. 制作数据集**

YOLO需要COCO类型数据，将VOC数据集转换为COCO数据集。

需要自定义修改的部分
- val_files_num: 验证集数量
- test_files_num: 测试集数量
- voc_annotations: 原始VOC数据集摆放位置中Annotations文件夹位置信息
- main_path: 原始VOC数据集的摆放位置

示例如下：

![image](https://github.com/He1senbergg/MidTerm-Part-II/assets/148076707/79c5979f-1268-4d36-9170-e59b8770b0d7)

下载repo中的voc_to_coco.py，放在根目录下，mmdetection/voc_to_coco.py。

命令行运行代码
```
python voc_to_coco.py
```

## Faster-R-CNN
**1. 训练**

使用Faster-R-CNN预训练权重进行训练。
```
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py
```

**2. 测试**

生成测试文件
```
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py epoch_8.pth --out results.pkl
```
挑选图像结果
```
python tools/analysis_tools/analyze_results_copy.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py results.pkl results --topk 20
```

**3. 推理**

在test/test_fasterrcnn.py，设置本地的pth（模型权重）、configs（配置信息）、picture（放入需要目标检测的图片）、result（目标检测结果）的位置。

运行命令行代码
```
python test/test_fasterrcnn.py
```

## YOLO V3
**1. 训练**
```python
python tools/train.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py
```

**2. 推理**

在test/test_yolo.py，设置本地的pth（模型权重）、configs（配置信息）、picture（放入需要目标检测的图片）、result（目标检测结果）的位置。

运行命令行代码
```
python test/test_yolo.py
```
