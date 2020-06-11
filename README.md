# MegEngine Models

![](https://github.com/MegEngine/Models/workflows/CI/badge.svg)

本仓库包含了采用[MegEngine](https://github.com/megengine/megengine)实现的各种主流深度学习模型。

[official](./official)目录下提供了各种经典的图像分类、目标检测、图像分割以及自然语言模型的官方实现。每个模型同时提供了模型定义、推理以及训练的代码。

官方会一直维护[official](./official)下的代码，保持适配MegEngine的最新API，提供最优的模型实现。同时，提供高质量的学习文档，帮助新手学习如何在MegEngine下训练自己的模型。

## 综述

对于每个模型，我们提供了至少四个脚本文件：模型定义(`model.py`)、模型推理(`inference.py`)、模型训练(`train.py`)、模型测试(`test.py`)。

每个模型目录下都对应有一个`README`，介绍了模型的详细信息，并详细描述了训练和测试的流程。例如 [ResNet README](./official/vision/classification/resnet/README.md)。

另外，`official`下定义的模型可以通过`megengine.hub`来直接加载，例如：

```bash
import megengine.hub

# 只加载网络结构
resnet18 = megengine.hub.load("megengine/models", "resnet18")
# 加载网络结构和预训练权重
resnet18 = megengine.hub.load("megengine/models", "resnet18", pretrained=True)
```

更多可以通过`megengine.hub`接口加载的模型见[hubconf.py](./hubconf.py)。

## 安装和环境配置

在开始运行本仓库下的代码之前，用户需要通过以下步骤来配置本地环境：

1. 克隆仓库

```bash
git clone https://github.com/MegEngine/Models.git
```

2. 安装依赖包

```bash
pip3 install --user -r requirements.txt
```

3. 添加目录到python环境变量中

```bash
export PYTHONPATH=/path/to/models:$PYTHONPATH
```


## 官方模型介绍

### 图像分类

图像分类是计算机视觉的基础任务。许多计算机视觉的其它任务（例如物体检测）都使用了基于图像分类的预训练模型。因此，我们提供了各种在ImageNet上预训练好的分类模型，包括[ResNet](./official/vision/classification/resnet)系列, [shufflenet](./official/vision/classification/shufflenet)系列等，这些模型在**ImageNet验证集**上的测试结果如下表：

| 模型 | top1 acc | top5 acc |
| :---: | :---: | :---: |
| ResNet18 |  70.312  |  89.430  |
| ResNet34 |  73.960  |  91.630  |
| ResNet50 | 76.254 | 93.056 |
| ResNet101 | 77.944 | 93.844 |
| ResNet152 | 78.582 | 94.130 | 
| ResNeXt50 32x4d | 77.592 | 93.644 |
| ResNeXt101 32x8d| 79.520 | 94.586 |
| ShuffleNetV2 x0.5 |  60.696  |  82.190  | 
| ShuffleNetV2 x1.0 |  69.372  |  88.764  | 
| ShuffleNetV2 x1.5 |  72.806  |  90.792  | 
| ShuffleNetV2 x2.0 |  75.074  |  92.278  | 


### 目标检测

目标检测同样是计算机视觉中的常见任务，我们提供了一个经典的目标检测模型[retinanet](./official/vision/detection)，这个模型在**COCO验证集**上的测试结果如下：

| 模型                       | mAP<br>@5-95  |
| :---:                        | :---:           |
| retinanet-res50-1x-800size | 36.0          |


### 图像分割

我们也提供了经典的语义分割模型--[Deeplabv3plus](./official/vision/segmentation/)，这个模型在**PASCAL VOC验证集**上的测试结果如下：

 |  模型       | Backbone    |  mIoU_single   | mIoU_multi  |
 |  :--:          |:--:     |:--:           |:--:         |
 |  Deeplabv3plus | Resnet101   | 79.0          | 79.8        |

<<<<<<< HEAD
<<<<<<< HEAD
### 人体关节点检测
=======
### 人体关节点
>>>>>>> update readme
=======
### 人体关节点
>>>>>>> update readme

我们提供了人体关节点检测的经典模型[SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf)和高精度模型[MSPN](https://arxiv.org/pdf/1901.00148.pdf)，使用在COCO val2017上人体检测AP为56的检测结果，提供的模型在COCO val2017上的关节点检测结果为:

|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
| SimpleBaseline |Res50 |256x192| 0.712 | 0.887 | 0.779 | 0.673 | 0.785 | 0.782 | 0.932 | 0.839 | 0.730 | 0.854 |
| SimpleBaseline |Res101|256x192| 0.722 | 0.891 | 0.795 | 0.687 | 0.795 | 0.794 | 0.936 | 0.855 | 0.745 | 0.863 |
| SimpleBaseline |Res152|256x192| 0.724 | 0.888 | 0.794 | 0.688 | 0.795 | 0.795 | 0.934 | 0.856 | 0.746 | 0.863 |
| MSPN_4stage |MSPN|256x192| 0.752 | 0.900 | 0.819 | 0.716 | 0.825 | 0.819 | 0.943 | 0.875 | 0.770 | 0.887 |

### 自然语言处理

我们同样支持一些常见的自然语言处理模型，模型的权重来自Google的pre-trained models, 用户可以直接使用`megengine.hub`轻松的调用预训练的bert模型。

另外，我们在[bert](./official/nlp/bert)中还提供了更加方便的脚本, 可以通过任务名直接获取到对应字典, 配置, 与预训练模型。

| 模型                       | 字典 | 配置 |
| ---                        |  --- |  --- |
| wwm_cased_L-24_H-1024_A-16| [link](https://data.megengine.org.cn/models/weights/bert/wwm_cased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/wwm_cased_L-24_H-1024_A-16/bert_config.json)
| wwm_uncased_L-24_H-1024_A-16| [link](https://data.megengine.org.cn/models/weights/bert/wwm_uncased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/wwm_uncased_L-24_H-1024_A-16/bert_config.json)
| cased_L-12_H-768_A-12| [link](https://data.megengine.org.cn/models/weights/bert/cased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/cased_L-12_H-768_A-12/bert_config.json)
| cased_L-24_H-1024_A-16| [link](https://data.megengine.org.cn/models/weights/bert/cased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/cased_L-24_H-1024_A-16/bert_config.json)
| uncased_L-12_H-768_A-12| [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-12_H-768_A-12/bert_config.json)
| uncased_L-24_H-1024_A-16| [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-24_H-1024_A-16/bert_config.json)
| chinese_L-12_H-768_A-12| [link](https://data.megengine.org.cn/models/weights/bert/chinese_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/chinese_L-12_H-768_A-12/bert_config.json)
| multi_cased_L-12_H-768_A-12| [link](https://data.megengine.org.cn/models/weights/bert/multi_cased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/multi_cased_L-12_H-768_A-12/bert_config.json)


在glue_data/MRPC数据集中使用默认的超参数进行微调和评估，评估结果介于84％和88％之间。

| Dataset | pretrained_bert | acc |
| --- |   --- |  --- |
| glue_data/MRPC |   uncased_L-12_H-768_A-12 |  86.25% |

