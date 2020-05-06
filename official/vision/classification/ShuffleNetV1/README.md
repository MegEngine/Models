# ShuffleNet Series

本目录包含了采用MegEngine实现的`ShuffleNet V1`网络结构，同时提供了在ImageNet训练集上的完整训练和测试代码。

`model.py`中定义了如下常见网络结构：`shufflenet_v1_x0_5_g3`, `shufflenet_v1_x0_5_g3_int8`

与原有网络结构不同，我们将ShuffleNetV1Block中需要下采样的Block中融合方式由Concat修改为Add

目前我们提供了部分在ImageNet上的预训练模型(见下表)，各个网络结构在ImageNet验证集上的表现如下：

| 模型 | float | int8 |
| --- | --- | --- |
| ShuffleNetV1 x0.5 (groups=3) |  56.87  |  55.73  | 
| ShuffleNetV1 x0.5 (groups=8) |    |    | 
| ShuffleNetV1 x1.0 (groups=3) |    |    | 
| ShuffleNetV1 x1.0 (groups=8) |    |    | 
| ShuffleNetV1 x1.5 (groups=3) |    |    | 
| ShuffleNetV1 x1.5 (groups=8) |    |    | 
| ShuffleNetV1 x2.0 (groups=3) |    |    | 
| ShuffleNetV1 x2.0 (groups=8) |    |    | 

用户可以通过`megengine.hub`直接加载本目录下定义好的模型，例如：

```bash
import megengine.hub

# 只加载网络结构
model = megengine.hub.load("megengine/models", "shufflenet_v1_x0_5")
# 加载网络结构和预训练权重
model = megengine.hub.load("megengine/models", "shufflenet_v1_x0_5", pretrained=True)
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

准备好数据集后，可以通过train.py开始训练：

`train.py`提供了灵活的命令行选项，包括：

- `--data`, ImageNet数据集的根目录，默认`/data/datasets/imagenet`;
- `--arch`, 需要训练的网络结构，默认`resnet18`；
- `--batch-size`，训练时每张卡采用的batch size, 默认128；
- `--ngpus`, 训练时采用的节点/gpu数量，默认1；当使用多张gpu时，将自动切换为分布式训练模式；
- `--save`, 模型以及log存储的目录，默认`/data/models`;
- `--learning-rate`, 训练时的初始学习率，默认0.0625，在分布式训练下，实际学习率等于初始学习率乘以节点/gpu数；
- `--steps`, 训练多少个iteration，默认150,000；
- `--quant`, 训练模式，normal：正常模型训练，qat：int8模型quantization aware training 

例如，可以通过以下命令在8块GPU上以128 x 8 = 1024的batch大小训练一个`shufflenet_v1_x0_5`的int8模型：

```bash
python3 train.py --data /path/to/imagenet \
                 --arch shufflenet_v1_x0_5 \
                 --batch-size 128 \
                 --learning-rate 0.0625 \
                 --ngpus 8 \
                 --save /path/to/save_dir \
                 
python3 train.py --data /path/to/imagenet \
                 --arch shufflenet_v1_x0_5_int8 \
                 --batch-size 128 \
                 --learning-rate 0.03125 \
                 --ngpus 8 \
                 --save /path/to/save_dir \
                 --model /path/to/pretrain                
```

更多详细的介绍可以通过运行`python3 train.py --help`查看。

## 如何测试

在训练的过程中，可以通过test.py测试模型在ImageNet验证集的性能

`test.py`的命令行选项如下：

- `--dataset-dir`，ImageNet数据集的根目录，默认`/data/datasets/imagenet`；
- `--arch`, 需要测试的网络结构，默认``；
- `--model`, 需要测试的模型；
- `--ngpus`, 用于测试的gpu数量，默认1；
- `--quantized`, 是否采用int8模型推理，仅支持CPU，并且--arch需设置为shufflenet_v1_x0_5_int8

全精度模型测试：
```bash
python3 test.py --dataset-dir=/path/to/imagenet --arch shufflenet_v1_x0_5 --model /path/to/model -ngpus 1
```

int8伪量化测试：
```bash
python3 test.py --dataset-dir=/path/to/imagenet --arch shufflenet_v1_x0_5_int8 --model /path/to/model -ngpus 1
```

int8测试：暂时仅支持CPU，速度很慢，不推荐
```bash
CUDA_VISIBLE_DEVICES=None python3 test.py --dataset-dir=/path/to/imagenet --arch shufflenet_v1_x0_5_int8 --model /path/to/model --quantized
```
更多详细介绍可以通过运行`python3 test.py --help`查看。

## 如何使用

模型训练好之后，可以通过inference.py测试单张图片:

`inference.py`的命令行选项如下：

- `--arch`, 需要使用的网络结构，默认`shufflenet_v1_x0_5`；
- `--model`, 训练好的模型权重地址；
- `--image`, 用于测试的图片；
- `--quantized`, 是否采用int8模型推理，仅支持CPU，并且--arch需设置为shufflenet_v1_x0_5

全精度模型测试：
```bash
python3 inference.py --arch shufflenet_v1_x0_5 --model /path/to/model --image /path/to/image
```

int8伪量化测试：
```bash
python3 inference.py --arch shufflenet_v1_x0_5_int8 --model /path/to/model --image /path/to/image
```

int8测试：暂时仅支持CPU
```bash
CUDA_VISIBLE_DEVICES=None python3 inference.py --arch shufflenet_v1_x0_5_int8 --model /path/to/model --image /path/to/image --quantized
```

模型量化训练完后，可以通过如下命令将其转化为int8推理模式的静态图：

```bash
CUDA_VISIBLE_DEVICES=None python3 convert.py --arch shufflenet_v1_x0_5_int8 --model /path/to/model
```
`convert.py`的命令行选项如下：

- `--arch`, 需要使用的网络结构，必须设置为以int8结尾的结构，默认`shufflenet_v1_x0_5_int8`；
- `--model`, 训练好的模型权重地址；
