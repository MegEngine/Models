# Semantic Segmentation

本目录包含了采用MegEngine实现的经典[Deeplabv3plus](https://arxiv.org/abs/1802.02611.pdf)网络结构，同时提供了在PASCAL VOC数据集上的完整训练和测试代码。

网络在PASCAL VOC2012验证集的性能和结果如下:

 Methods       | Backbone    | TrainSet  | EvalSet | mIoU_single   | mIoU_multi  |
 :--:          |:--:         |:--:       |:--:     |:--:           |:--:         |
 Deeplabv3plus | Resnet101   | train_aug | val     | 79.0          | 79.8        |


## 安装和环境配置

在开始运行本目录下的代码之前，请确保按照[README](../../../../README.md)进行了正确的环境配置。


## 如何训练

1、在开始训练前，请下载[VOC2012官方数据集](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data)，并解压到合适的目录下。为保证一样的训练环境，还需要下载[SegmentationClassAug](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0&file_subpath=%2FSegmentationClassAug)。具体可以参照这个[流程](https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/)。

准备好的 VOC 数据目录结构如下：

```bash
/path/to/
    |->VOC2012
    |    |Annotations
    |    |ImageSets
    |    |JPEGImages
    |    |SegmentationClass
    |    |SegmentationClass_aug
```
其中，ImageSets/Segmentation中包含了[trainaug.txt](https://gist.githubusercontent.com/sun11/2dbda6b31acc7c6292d14a872d0c90b7/raw/5f5a5270089239ef2f6b65b1cc55208355b5acca/trainaug.txt)。

注意：SegmentationClass_aug和SegmentationClass中的数据格式不同。

2、准备好预训练好的backbone权重，可以直接下载megengine官方提供的在ImageNet上预训练的resnet101模型。

3、开始训练:

`train.py`的命令行参数如下：
- `--dataset_dir`，训练时采用的训练集存放的目录;
- `--weight_file`，训练时采用的预训练权重;
- `--batch-size`，训练时采用的batch size, 默认8；
- `--ngpus`, 训练时采用的gpu数量，默认8; 当设置为1时，表示单卡训练
- `--resume`, 是否从已训好的模型继续训练；
- `--train_epochs`, 需要训练的epoch数量；

```bash
python3 train.py --dataset_dir /path/to/VOC2012 \
                 --weight_file /path/to/weights.pkl \
                 --batch_size 8 \
                 --ngpus 8 \
                 --train_epochs 50 \
                 --resume /path/to/model
```

## 如何测试

模型训练好之后，可以通过如下命令测试模型在VOC2012验证集的性能：

```bash
python3 test.py --dataset_dir /path/to/VOC2012 \
                --model_path /path/to/model.pkl
```

`test.py`的命令行参数如下：
- `--dataset_dir`，验证时采用的验证集目录;
- `--model_path`，载入训练好的模型；

## 如何使用

模型训练好之后，可以通过如下命令测试单张图片，得到分割结果：

```bash
python3 inference.py --model_path /path/to/model \
                     --image_path /path/to/image.jpg
```

`inference.py`的命令行参数如下：
- `--model_path`，载入训练好的模型；
- `--image_path`，载入待测试的图像

<div align="left">
<img src="../../assets/cat.jpg" height="500px" alt="input" ><img src="../../assets/cat_seg_out.jpg" height="500px" alt="output" >
</div>

## 参考文献

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611.pdf), Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and
Hartwig Adam; ECCV, 2018
