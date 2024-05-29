# 一、Repo内容说明
**1. 任务一**

内容为“任务一/”下所示的四个python文件。

**2. 任务二**

其余文件均为任务二相关文件。

# 二、任务一
## Ⅰ. 准备步骤
**1. 代码下载**

下载“任务一/”下的四个python文件，放在同级目录

调整终端目录，以便train.py能方便的导入其他同级目录的函数。

命令行运行代码
```
cd 四个文件摆放的同级目录位置
```

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

**4. 下载模型权重文件**

该模型权重，是在pre-trained的ResNet-18基础上微调得到的结果。
```
wget https://www.dropbox.com/scl/fi/60rnqnk25st3p0f72163t/23_0.7041767345529858.pth?rlkey=k4fahzd92cn8chax1br9ynuvo&st=i83qjicl&dl=1
```

## Ⅱ. 训练

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
- 示例4（请注意修改以下的信息的绝对位置）：使用本地模型pth与其他默认参数开始训练
  ```
  python train.py --data_dir /data/CUB_200_2011 --save_dir /model --logdir /tensorboard --model_path model.pth
  ```

## Ⅲ. 测试

测试的效果为：输出如下信息
```python
print(f'Test Loss: {val_loss:.4f}, Test Accuracy: {val_acc:.4f}, Test Time: {val_elapsed_time:.2f}s')
```

测试时，需要提供两个参数`--data_dir`、`--model_path`。

命令行运行代码，示例如下（请注意修改以下的信息的绝对位置）：
```
python test.py --data_dir /data/CUB_200_2011 --model_path model.pth
```

# 三、任务二
## Ⅰ. 准备步骤
**1. 调整配置文件**

按照Repo中的相对位置，将需要修改的配置放入mmdetection文件夹中。

*请注意，其中`analyze_results.py`，是在文件夹mmdetection/tools/analysis_tools/analyze_results.py*

接着，调整目录

命令行运行代码
```
cd mmdetection
```

**2. 下载模型权重**

- Faster-R-CNN预训练权重（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，或直接以绝对位置执行。）：
```python
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth -P checkpoints/
```
- 本地服务器训练以后的Faster-R-CNN模型权重文件
  - dropbox下载
  ```
  wget https://www.dropbox.com/scl/fi/sgtkp0by8p7f4u4c0fh0i/epoch_8.pth?rlkey=zmvpwjce9anpqx6o03gvubvi3&st=arx3wpfi&dl=1
  ```

- 本地服务器训练以后的YOLO V3模型权重文件
  - dropbox下载
  ```
  wget https://www.dropbox.com/scl/fi/qa7ziesle7kxfunojsxcc/epoch_273.pth?rlkey=mopji9d302b0q7gh234v73weg&st=h1i1z1nc&dl=1
  ```

**3. 制作数据集**

YOLO需要COCO类型数据，将VOC数据集转换为COCO数据集。作业中，只转化了2012数据集用来使用。

需要自定义修改的部分
- val_files_num: 验证集数量
- test_files_num: 测试集数量
- voc_annotations: 原始VOC数据集摆放位置中Annotations文件夹位置信息
- main_path: 原始VOC数据集的摆放位置

示例如下：

![image](https://github.com/He1senbergg/MidTerm-Part-II/assets/148076707/fe766601-c877-4c94-a00e-f3ba59acdc1e)

下载repo中的voc_to_coco.py。

命令行运行代码（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，该代码运行需要相对位置成立。）
```
python voc_to_coco.py
```

## Ⅱ. Faster-R-CNN
**1. 训练**

使用Faster-R-CNN预训练权重进行训练。

命令行运行代码（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，或直接以绝对位置执行。）
```
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py
```

**2. 测试VOC数据集**

生成测试文件

命令行运行代码（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，或直接以绝对位置执行。）
```
python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py epoch_8.pth --out results.pkl
```
挑选图像结果

命令行运行代码（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，或直接以绝对位置执行。）
```
python tools/analysis_tools/analyze_results_copy.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py results.pkl results --topk 20
```

**3. 推理数据集外图片**

在test/test.py，设置本地的`--fasterpth`（模型权重）、`--fasterconfig`（配置信息）、`--imgfolder`（放入需要目标检测的图片）、`--output`（目标检测结果）的位置。

或直接如下，在命令行中输入这四个参数的信息。

命令行运行代码（请注意修改以下的信息的绝对位置）
```
python test/test.py --fasterpth '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/faster-rcnn/1/epoch_8.pth' --fasterconfig "/mnt/ly/models/mmdetection/mmdetection-main/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py" --imgfolder "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/1" --output "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/result"
```

- 同时，可以设置`--bbox_color`（锚框颜色）、`--text_color`（类别字体颜色）。

  示例代码（该代码只是说明颜色的便捷设置途径，真正执行需仿照如上的代码，提供其余信息）：
  ```
  python test/test.py --bbox_color white --text_color red
  ```
- 并且我修改了from mmcv.visualization.image import imshow_det_bboxes
  在此处，给函数补上了参数`thickness=2`，以加粗字体。
  ```
     cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, thickness=2)
  ```
  
## Ⅲ. YOLO V3
**1. 训练**

命令行运行代码（此处请确保已经执行“三、Ⅰ.1.”中的切换主目录，或直接以绝对位置执行。）
```python
python tools/train.py configs/yolo/yolov3_d53_8xb8-320-273e_coco.py
```

**2. 推理数据集外图片**

在test/test.py，设置本地的`--yolopth`（模型权重）、`--yoloconfig`（配置信息）、`--imgfolder`（放入需要目标检测的图片）、`--output`（目标检测结果）的位置。

或直接如下，在命令行中输入这四个参数的信息。
  
命令行运行代码（请注意修改以下的信息的绝对位置）
```
python test/test.py --yolopth '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/3/epoch_273.pth' --yoloconfig "/mnt/ly/models/mmdetection/mmdetection-main/configs/yolo/yolov3_d53_8xb8-320-273e_coco.py" --imgfolder "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/1" --output "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/result"
```

- 同时，可以设置`--bbox_color`（锚框颜色）、`--text_color`（类别字体颜色）。

  示例代码（该代码只是说明颜色的便捷设置途径，真正执行需仿照如上的代码，提供其余信息）：
  ```
  python test/test.py --bbox_color white --text_color red
  ```
- 并且我修改了from mmcv.visualization.image import imshow_det_bboxes
  在此处，给函数补上了参数`thickness=2`，以加粗字体。
  ```
     cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, thickness=2)
  ```
  
## Ⅳ. 对比Faster-R-CNN与YOLO V3
在test/test.py，设置本地的`--compare`（是否进行对比，即两个模型的结果通过plt放在一行中展示）、`--fasterpth`（模型权重）、`--fasterconfig`（配置信息）、`--yolopth`（模型权重）、`--yoloconfig`（配置信息）、`--imgfolder`（放入需要目标检测的图片）、`--output`（目标检测结果）的位置。

或直接如下，在命令行中输入这四个参数的信息。
  
命令行运行代码（请注意修改以下的信息的绝对位置）
```
python test/test.py --compare True --fasterpth '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/faster-rcnn/1/epoch_8.pth' --fasterconfig "/mnt/ly/models/mmdetection/mmdetection-main/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712.py" --yolopth '/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/3/epoch_273.pth' --yoloconfig "/mnt/ly/models/mmdetection/mmdetection-main/configs/yolo/yolov3_d53_8xb8-320-273e_coco.py" --imgfolder "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/1" --output "/mnt/ly/models/mmdetection/mmdetection-main/work_dirs/cfm/yolov3/test_1/out_picture/result/Comparison"
```

- 同时，可以设置`--bbox_color`（锚框颜色）、`--text_color`（类别字体颜色）。

  示例代码（该代码只是说明颜色的便捷设置途径，真正执行需仿照如上的代码，提供其余信息）：
  ```
  python test/test.py --bbox_color white --text_color red
  ```
- 并且我修改了from mmcv.visualization.image import imshow_det_bboxes
  在此处，给函数补上了参数`thickness=2`，以加粗字体。
  ```
     cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color, thickness=2)
  ```
