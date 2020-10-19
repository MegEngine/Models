# ShuffleNet Series

本目录包含了采用MegEngine实现的`ShuffleNet V2`网络结构，同时提供了在ImageNet训练集上的完整训练和测试代码。

`model.py`中定义了如下常见网络结构：`shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0`.

目前我们提供了部分在ImageNet上的预训练模型(见下表)，各个网络结构在ImageNet验证集上的表现如下：

| 模型 | top1 acc | top5 acc |
| --- | --- | --- |
| ShuffleNetV2 x0.5 |  60.696  |  82.190  |
| ShuffleNetV2 x1.0 |  69.372  |  88.764  |
| ShuffleNetV2 x1.5 |  72.806  |  90.792  |
| ShuffleNetV2 x2.0 |  75.074  |  92.278  |

用户可以通过`megengine.hub`直接加载本目录下定义好的模型，例如：

```bash
import megengine.hub

# 只加载网络结构
resnet18 = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0")
# 加载网络结构和预训练权重
resnet18 = megengine.hub.load("megengine/models", "shufflenet_v2_x1_0", pretrained=True)
```

## 安装和环境配置

在开始运行本目录下的代码之前，请确保按照[README](../../../../README.md)进行了正确的环境配置。

## 如何训练

在开始训练前，请确保已经下载解压好[ImageNet数据集](http://image-net.org/download)，并放在合适的目录下，准备好的数据集的目录结构如下所示：

```bash
/path/to/imagenet
    train
         n01440764
              xxx.jpg
              ...
         n01443537
              xxx.jpg
              ...
         ...
    val
         n01440764
              xxx.jpg
              ...
         n01443537
              xxx.jpg
              ...
         ...
```

准备好数据集后，可以运行以下命令开始训练：

```bash
python3 train.py --data=/path/to/imagenet
```

`train.py`提供了灵活的命令行选项，包括：

- `--data`, ImageNet数据集的根目录，默认`/data/datasets/imagenet`;
- `--arch`, 需要训练的网络结构，默认`shufflenet_v2_x1_0`；
- `--batch-size`，训练时每张卡采用的batch size, 默认128；
- `--ngpus`, 训练时每个节点采用的gpu数量，默认`None`，即使用全部gpu；当使用多张gpu时，将自动切换为分布式训练模式；
- `--save`, 模型以及log存储的目录，默认`outputs`;
- `--learning-rate`, 训练时的初始学习率，默认0.0625，在分布式训练下，实际学习率等于初始学习率乘以节点/gpu数；
- `--epochs`, 训练多少个epoch，默认240；

例如，可以通过以下命令在8块GPU上以128 x 8 = 1024的batch大小训练一个`shufflenet_v2_x1_5`的模型：

```bash
python3 train.py --data /path/to/imagenet \
                 --arch shufflenet_v2_x1_0 \
                 --batch-size 128 \
                 --learning-rate 0.0625 \
                 --ngpus 8 \
                 --save /path/to/save_dir
```

更多详细的介绍可以通过运行`python3 train.py --help`查看。

## 如何测试

在训练的过程中，可以通过如下命令测试模型在ImageNet验证集的性能：

```bash
python3 test.py --data /path/to/imagenet --arch shufflenet_v2_x1_0 --model /path/to/model --ngpus 1
```

`test.py`的命令行选项如下：

- `--data`，ImageNet数据集的根目录，默认`/data/datasets/imagenet`；
- `--arch`, 需要测试的网络结构，默认``；
- `--model`, 需要测试的模型，默认使用官方预训练模型；
- `--ngpus`, 用于测试的gpu数量，默认`None`；

更多详细介绍可以通过运行`python3 test.py --help`查看。

## 如何使用

模型训练好之后，可以通过如下命令测试单张图片:

```bash
python3 inference.py --model /path/to/model --image /path/to/image.jpg
```

使用默认的测试图片和默认的`shufflenet_v2_x1_0`预训练模型，将输出如下结果：
```
0: class = Siamese_cat          with probability = 53.5 %
1: class = lynx                 with probability =  6.9 %
2: class = tabby                with probability =  4.6 %
3: class = Persian_cat          with probability =  2.6 %
4: class = Angora               with probability =  1.4 %
```

`inference.py`的命令行选项如下：

- `--arch`, 需要使用的网络结构，默认`shufflenet_v2_x1_0`；
- `--model`, 训练好的模型权重地址，默认使用官方预训练的`shufflenet_v2_x1_0`模型；
- `--image`, 用于测试的图片；

## 参考文献

- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164), Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
