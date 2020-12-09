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

图像分类是计算机视觉的基础任务。许多计算机视觉的其它任务（例如物体检测）都使用了基于图像分类的预训练模型。因此，我们提供了各种在ImageNet上预训练好的分类模型，
具体实现模型参考[这里](./official/vision/classification).

### 目标检测

目标检测同样是计算机视觉中的常见任务，我们提供了多个经典的目标检测模型，具体模型的实现可以参考[这里](./official/vision/detection).

### 图像分割

语意分割也是计算机视觉中的一项基础任务，为此我们也提供了经典的语义分割模型，具体可以参考[这里](./official/vision/segmentation/).

### 人体关节点检测

我们提供了人体关节点检测的经典模型和高精度模型，具体的实现可以参考[这里](./official/vision/keypoints).

### 自然语言处理

我们同样支持一些常见的自然语言处理模型，模型的权重来自Google的pre-trained models, 用户可以直接使用`megengine.hub`轻松的调用预训练的bert模型。

另外，我们在[bert](./official/nlp/bert)中还提供了更加方便的脚本, 可以通过任务名直接获取到对应字典, 配置, 与预训练模型。
