# Human Pose Esimation

本目录包含了采用MegEngine实现的经典[SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf)和[MSPN](https://arxiv.org/pdf/1901.00148.pdf)网络结构，同时提供了在COCO数据集上的完整训练和测试代码。

本目录使用了在COCO val2017上的Human AP为56.4的人体检测结果，最后在COCO val2017上人体关节点估计结果为
|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
| SimpleBaseline |Res50 |256x192| 0.712 | 0.887 | 0.779 | 0.673 | 0.785 | 0.782 | 0.932 | 0.839 | 0.730 | 0.854 |
| SimpleBaseline |Res101|256x192| 0.722 | 0.891 | 0.795 | 0.687 | 0.795 | 0.794 | 0.936 | 0.855 | 0.745 | 0.863 |
| SimpleBaseline |Res152|256x192| 0.724 | 0.888 | 0.794 | 0.688 | 0.795 | 0.795 | 0.934 | 0.856 | 0.746 | 0.863 |
| MSPN_4stage |MSPN|256x192| 0.752 | 0.900 | 0.819 | 0.716 | 0.825 | 0.819 | 0.943 | 0.875 | 0.770 | 0.887 |

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
|-- |-- train2017
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

更改[config.py](.config.py)中的`data_root`为${COCO_DATA_ROOT}

3、开始训练:

`train.py`的命令行参数如下:
- `--arch`, 训练的网络的名字
- `--resume`, 是否从已训好的模型继续训练
- `--ngpus`, 使用的GPU数量
- `--multi_scale_supervision`, 是否使用多尺度监督；

例如训练SimpleBaseline_Res50:
```bash
python3 train.py --arch simplebaseline_res50 \
                 --resume /path/to/model \
                 --ngpus 8 \
                 --multi_scale_supervision False
                 
```
训练MSPN:
```bash
python3 train.py --arch mspn_4stage \
                 --resume /path/to/model \
                 --ngpus 8 \
                 --multi_scale_supervision True

```

## 如何测试

模型训练好之后，可以通过如下命令测试模型在COCOval2017验证集的性能：

```bash
python3 test.py --arch name/of/network \
                --model /path/to/model.pkl \
                --dt_file /name/human/detection/results
```

`test.py`的命令行参数如下：
- `--arch`, 网络的名字;
- `--model`, 待检测的模;
- `--dt_path`，人体检测结果.

## 如何使用

模型训练好之后，可以通过如下命令测试单张图片(先使用预训练的RetainNet检测出人的框），得到人体姿态可视化结果：

```bash
python3 inference.py --arch /name/of/tested/network \
                     --detector /name/of/human/detector \
                     --model /path/to/model \
                     --image /path/to/image.jpg
```

`inference.py`的命令行参数如下：
- `--arch`, 网络的名字;
- `--detector`, 人体检测器的名字;
- `--model`，载入训练好的模型;
- `--image`，载入待测试的图像.

## 参考文献

- [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/pdf/1804.06208.pdf), Bin Xiao, Haiping Wu, and Yichen Wei
- [Rethinking on Multi-Stage Networks for Human Pose Estimation](https://arxiv.org/pdf/1901.00148.pdf) Wenbo Li1, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei and Jian Sun
