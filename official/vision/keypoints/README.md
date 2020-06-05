# Human Pose Esimation

本目录包含了采用MegEngine实现的经典[SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf)和[MSPN](https://arxiv.org/pdf/1901.00148.pdf)网络结构，同时提供了在COCO数据集上的完整训练和测试代码。

本目录使用了在COCO val2017上的Human AP为56.4的人体检测结果，最后在COCO val2017上人体关节点估计结果为
|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
<<<<<<< HEAD
| SimpleBaseline |Res50 |256x192| 71.2 | 0.887 | 0.779 | 0.673 | 0.785 | 0.782 | 0.932 | 0.839 | 0.730 | 0.854 |
| SimpleBaseline |Res101|256x192| 72.2 | 0.891 | 0.795 | 0.687 | 0.795 | 0.794 | 0.936 | 0.855 | 0.745 | 0.863 |
| SimpleBaseline |Res152|256x192| 72.4 | 0.888 | 0.794 | 0.688 | 0.795 | 0.795 | 0.934 | 0.856 | 0.746 | 0.863 |
=======
| SimpleBaseline |Res50|256x192| 70.4 |0.886 | 0.783 | 0.671 | 0.772 | 0.763 | 0.929 | 0.834 | 0.721 | 0.824 |
>>>>>>> add readme


## 安装和环境配置

* 在开始运行本目录下的代码之前，请确保按照[README](../../../../README.md)进行了正确的环境配置。
* 安装[COCOAPI](https://github.com/cocodataset/cocoapi):
```bash
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
```


## 如何训练

1、在开始训练前，请下载[COCO官方数据集](http://cocodataset.org/#download)，并解压到合适的目录下。从[OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) 或者 [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing)下载COCO val2017上人体检测的结果，该结果在COCO val2017上人体检测AP为56.

准备好的 COCO 数据目录结构如下：
```bash
${COCO_DATA_ROOT}
|-- annotations
|   |-- person_keypoints_train2017.json
|   |-- person_keypoints_val2017.json
|-- person_detection_results
|   |-- COCO_val2017_detections_AP_H_56_person.json
|-- images
    |-- train2017
    |   |-- 000000000009.jpg
    |   |-- 000000000025.jpg
    |   |-- 000000000030.jpg
    |   |-- ... 
    |-- val2017
        |-- 000000000139.jpg
        |-- 000000000285.jpg
        |-- 000000000632.jpg
        |-- ... 
```


3、开始训练:

<<<<<<< HEAD
`train.py`的命令行参数如下:
- `--arch`, 训练的模型的名字
=======
`train.py`的命令行参数如下：
>>>>>>> add readme
- `--data_root`，COCO数据集里`images`的路径；
- `--ann_file`, COCO数据集里标注文件的`json`路径
- `--batch_size`，训练时采用的batch size, 默认32；
- `--ngpus`, 训练时采用的gpu数量，默认8; 当设置为1时，表示单卡训练
- `--continue`, 是否从已训好的模型继续训练；
<<<<<<< HEAD
- `--epochs`, 需要训练的epoch数量；
- `--lr`, 初始学习率；

```bash
python3 train.py --arch name/of/model \
                 --data_root /path/to/COCO/images \
=======
- `--train_epochs`, 需要训练的epoch数量；
- `--lr`, 初始学习率；

```bash
python3 train.py --data_root /path/to/COCO images \
>>>>>>> add readme
                 --ann_file /path/to/person_keypoints.json \
                 --batch_size 32 \
                 --lr 0.0003 \
                 --ngpus 8 \
<<<<<<< HEAD
                 --epochs 200 \
                 --continue /path/to/model
=======
                 --train_epochs 200 \
                 --resume /path/to/model
>>>>>>> add readme
```

## 如何测试

<<<<<<< HEAD
模型训练好之后，可以通过如下命令测试模型在COCOval2017验证集的性能：

```bash
python3 test.py --arch name/of/model \
                --data_root /path/to/COCO/images \
=======
先准备好人体检测的
模型训练好之后，可以通过如下命令测试模型在COCOval2017验证集的性能：

```bash
python3 test.py --data_root /path/to/COCO/images \
>>>>>>> add readme
                --model /path/to/model.pkl \
                --gt_path /path/to/ground/truth/annotations
                --dt_path /path/to/human/detection/results
```

`test.py`的命令行参数如下：
<<<<<<< HEAD
- `--arch`, 训练的模型的名字
- `--data_root`，COCO数据集里`images`的路径;
- `--gt_path`, COCO数据集里验证集的标注文件;
=======
- `--data_root`，COCO数据集里`images`的路径;
- `--gt_path`, COCO数据集里验证集的标注文件
>>>>>>> add readme
- `--dt_path`，人体检测结果；
- `--model`, 待检测的模型

## 如何使用

<<<<<<< HEAD
模型训练好之后，可以通过如下命令测试单张图片(先使用预训练的RetainNet检测出人的框），得到人体姿态可视化结果：

```bash
python3 inference.py --arch /name/of/tested/model \
                     --model /path/to/model \
                     --image /path/to/image.jpg
```

`inference.py`的命令行参数如下：
- `--arch`, 网络的名字;
- `--model`，载入训练好的模型;
- `--image`，载入待测试的图像
=======
模型训练好之后，可以通过如下命令测试单张图片，得到分割结果：

```bash
python3 inference.py --model_path /path/to/model \
                     --image_path /path/to/image.jpg
```

`inference.py`的命令行参数如下：
- `--model_path`，载入训练好的模型；
- `--image_path`，载入待测试的图像
>>>>>>> add readme

## 参考文献

- [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/pdf/1804.06208.pdf), Bin Xiao, Haiping Wu, and Yichen Wei
- [Rethinking on Multi-Stage Networks for Human Pose Estimation](https://arxiv.org/pdf/1901.00148.pdf) Wenbo Li1, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei and Jian Sun

