# Megengine RetinaNet

## 介绍

本目录包含了采用MegEngine实现的经典[RetinaNet](https://arxiv.org/pdf/1708.02002>)网络结构，同时提供了在COCO2017数据集上的完整训练和测试代码。

网络的性能在COCO2017验证集上的测试结果如下：

| 模型                            | mAP<br>@5-95 | batch<br>/gpu | gpu    | speed<br>(8gpu)   | speed<br>(1gpu) |
| ---                             | ---          | ---           | ---    | ---               | ---             |
| retinanet-res50-coco-1x-800size | 36.0         | 2             | 2080ti | 2.27(it/s)        | 3.7(it/s)       |

* MegEngine v0.4.0

## 如何使用

模型训练好之后，可以通过如下命令测试单张图片:

```bash
python3 tools/inference.py -f retinanet_res50_coco_1x_800size.py \
                           -i ../../assets/cat.jpg \
                           -m /path/to/retinanet_weights.pkl
```

`tools/inference.py`的命令行选项如下:

- `-f`, 测试的网络结构描述文件。
- `-m`, 网络结构文件所对应的训练权重, 可以从顶部的表格中下载训练好的检测器权重。
- `-i`, 需要测试的样例图片。

使用默认图片和默认模型测试的结果见下图:

![demo image](../../assets/cat_det_out.jpg)

## 如何训练

1. 在开始训练前，请确保已经下载解压好[COCO2017数据集](http://cocodataset.org/#download)，
并放在合适的数据目录下，准备好的数据集的目录结构如下所示(目前默认使用COCO2017数据集)：

```
/path/to/
    |->coco
    |    |annotations
    |    |train2017
    |    |val2017
```

2. 准备预训练的`backbone`网络权重：可使用 megengine.hub 下载`megengine`官方提供的在ImageNet上训练的ResNet-50模型, 并存放在 `/path/to/pretrain.pkl`。

3. 在开始运行本目录下的代码之前，请确保按照[README](../../../README.md)进行了正确的环境配置。

4. 开始训练:

```bash
python3 tools/train.py -f retinanet_res50_coco_1x_800size.py \
                       -n 8 \
                       --batch_size 2 \
                       -w /path/to/pretrain.pkl
```

`tools/train.py`提供了灵活的命令行选项，包括：

- `-f`, 所需要训练的网络结构描述文件。
- `-n`, 用于训练的devices(gpu)数量，默认使用所有可用的gpu.
- `-w`, 预训练的backbone网络权重的路径。
- `--batch_size`，训练时采用的`batch size`, 默认2，表示每张卡训2张图。
- `--dataset-dir`, COCO2017数据集的上级目录，默认`/data/datasets`。

默认情况下模型会存在 `log-of-retinanet_res50_1x_800size`目录下。

## 如何测试

在训练的过程中，可以通过如下命令测试模型在`COCO2017`验证集的性能：

```bash
python3 tools/test.py -f retinanet_res50_coco_1x_800size.py \
                      -n 8 \
                      --model /path/to/retinanet_weights.pt \
                      --dataset_dir /data/datasets
```

`tools/test.py`的命令行选项如下：

- `-f`, 所需要测试的网络结构描述文件。
- `-n`, 用于测试的devices(gpu)数量，默认1；
- `--model`, 需要测试的模型；可以从顶部的表格中下载训练好的检测器权重, 也可以用自行训练好的权重。
- `--dataset_dir`，COCO2017数据集的上级目录，默认`/data/datasets`

## 参考文献

- [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002) Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf)  Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence
Lin T Y, Maire M, Belongie S, et al. European conference on computer vision. Springer, Cham, 2014: 740-755.
