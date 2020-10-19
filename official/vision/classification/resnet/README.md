# ResNet Series

本目录包含了采用MegEngine实现的经典`ResNet`网络结构，同时提供了在ImageNet训练集上的完整训练和测试代码。

`model.py`中定义了一些常见的网络结构：`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`等.

目前我们提供了部分在ImageNet上的预训练模型(见下表)，各个网络结构在ImageNet验证集上的测试结果如下：

| 模型 | top1 acc | top5 acc |
| --- | --- | --- |
| ResNet18 |  70.312  |  89.430  |
| ResNet34 |  73.960  |  91.630  |
| ResNet50 | 76.254 | 93.056 |
| ResNet101 | 77.944 | 93.844 |
| ResNet152 | 78.582 | 94.130 |
| ResNeXt50 32x4d | 77.592 | 93.644 |
| ResNeXt101 32x8d| 79.520 | 94.586 |

用户可以通过`megengine.hub`直接加载本目录下定义好的模型，例如：

```bash
import megengine.hub

# 只加载网络结构
resnet18 = megengine.hub.load("megengine/models", "resnet18")
# 加载网络结构和预训练权重
resnet18 = megengine.hub.load("megengine/models", "resnet18", pretrained=True)
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
python3 train.py --dataset-dir=/path/to/imagenet
```

`train.py`提供了灵活的命令行选项，包括：

- `--data`, ImageNet数据集的根目录，默认`/data/datasets/imagenet`;
- `--arch`, 需要训练的网络结构，默认`resnet50`；
- `--batch-size`，训练时每张卡采用的batch size, 默认64；
- `--ngpus`, 训练时每个节点采用的gpu数量，默认`None`，即使用全部gpu；当使用多张gpu时，将自动切换为分布式训练模式；
- `--save`, 模型以及log存储的目录，默认`output`;
- `--learning-rate`, 训练时的初始学习率，默认0.025，在分布式训练下，实际学习率等于初始学习率乘以总gpu数；
- `--epochs`, 训练多少个epoch，默认90；

例如，可以通过以下命令在2块GPU上以64的batch大小训练一个`resnet50`的模型：

```bash
python3 train.py --data /path/to/imagenet \
                 --arch resnet50 \
                 --batch-size 64 \
                 --learning-rate 0.025 \
                 --ngpus 2 \
                 --save /path/to/save_dir
```

更多详细的介绍可以通过运行`python3 train.py --help`查看。

## 如何测试

在训练的过程中，可以通过如下命令测试模型在ImageNet验证集的性能：

```bash
python3 test.py --data=/path/to/imagenet --arch resnet50 --model /path/to/model --ngpus 1
```

`test.py`的命令行选项如下：

- `--data`，ImageNet数据集的根目录，默认`/data/datasets/imagenet`；
- `--arch`, 需要测试的网络结构，默认`resnet50`；
- `--model`, 需要测试的模型，默认使用官方预训练模型；
- `--ngpus`, 用于测试的gpu数量，默认`None`；

更多详细介绍可以通过运行`python3 test.py --help`查看。

## 如何使用

模型训练好之后，可以通过如下命令测试单张图片:

```bash
python3 inference.py --model /path/to/model --image /path/to/image.jpg
```

使用默认的测试图片和默认的resnet18模型，将输出如下结果：
```
0: class = lynx                 with probability = 25.2 %
1: class = Siamese_cat          with probability = 12.3 %
2: class = Egyptian_cat         with probability =  8.7 %
3: class = Persian_cat          with probability =  8.3 %
4: class = tabby                with probability =  6.5 %
```

`inference.py`的命令行选项如下：

- `--arch`, 需要使用的网络结构，默认`resnet18`；
- `--model`, 训练好的模型权重地址，默认使用官方预训练的resnet18模型；
- `--image`, 用于测试的图片；

## 参考文献

- [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778
- [Aggregated Residual Transformation for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf), Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He; The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1492-1500
- [Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf), Sergey Zagoruyko, Nikos Komodakis, arXiv:1605.07146
