# Human Pose Esimation

本目录包含了采用MegEngine实现的经典[SimpleBaseline](https://arxiv.org/pdf/1804.06208.pdf), [HRNet](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf), [MSPN](https://arxiv.org/pdf/1901.00148.pdf)和[RSN](https://arxiv.org/pdf/2003.04030.pdf)的网络结构，同时提供了在COCO数据集上的完整训练和测试代码。

本目录使用了在COCO val2017上的Human AP为56.4的人体检测结果，最后在COCO val2017上人体关节点估计结果为
|Methods|Backbone|Input Size| AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|:---:|---|---|---|---|---|---|---|---|---|---|---|
| SimpleBaseline |Res50 |256x192| 0.712 | 0.887 | 0.779 | 0.673 | 0.785 | 0.782 | 0.932 | 0.839 | 0.730 | 0.854 |
| SimpleBaseline |Res101|256x192| 0.722 | 0.891 | 0.795 | 0.687 | 0.795 | 0.794 | 0.936 | 0.855 | 0.745 | 0.863 |
| SimpleBaseline |Res152|256x192| 0.724 | 0.888 | 0.794 | 0.688 | 0.795 | 0.795 | 0.934 | 0.856 | 0.746 | 0.863 |
| HRNet-W32   |HRNet-W32|256x192| 0.740 | 0.896 | 0.806 | 0.702 | 0.812 | 0.805 | 0.938 | 0.860 | 0.756 | 0.871 |
| HRNet-W48   |HRNet-W48|256x192| 0.745 | 0.899 | 0.811 | 0.706 | 0.818 | 0.810 | 0.942 | 0.867 | 0.760 | 0.878 |
| RSN18       |RSN18    |256x192| 0.721 | 0.891 | 0.789 | 0.682 | 0.795 | 0.790 | 0.935 | 0.846 | 0.738 | 0.860 |
| RSN50       |RSN50    |256x192| 0.732 | 0.895 | 0.800 | 0.692 | 0.805 | 0.799 | 0.937 | 0.856 | 0.748 | 0.868 |
| MSPN_4stage |MSPN     |256x192| 0.752 | 0.900 | 0.819 | 0.716 | 0.825 | 0.819 | 0.943 | 0.875 | 0.770 | 0.887 |

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
|-- train2017
|   |   |-- 000000000009.jpg
|   |   |-- 000000000025.jpg
|   |   |-- 000000000030.jpg
|   |   |-- ... 
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

模型训练好之后，可以通过如下命令测试指定模型在COCOval2017验证集的性能：
```bash
python3 test.py --arch name/of/network \
                --model /path/to/model.pkl \
                --dt_file /name/human/detection/results
```
`test.py`的命令行参数如下：
- `--arch`, 网络的名字;
- `--model`, 待检测的模;
- `--dt_path`，人体检测结果.

也可以连续验证多个模型的性能:

```bash
python3 test.py --arch name/of/network \
                --model_dir path/of/saved/models \
                --start_epoch num/of/start/epoch \
                --end_epoch num/of/end/epoch \
                --test_freq test/frequence
```

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

- [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) Bin Xiao, Haiping Wu, and Yichen Wei. European Conference on Computer Vision (ECCV), 2018.
- [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212) Ke Sun Bin Xiao Dong Liu Jingdong Wang. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
- [Learning Delicate Local Representations for Multi-Person Pose Estimation](https://arxiv.org/abs/2003.04030) Yuanhao Cai, Zhicheng Wang, Zhengxiong Luo, Binyi Yin, Angang Du, Haoqian Wang, Xiangyu Zhang, Xinyu Zhou, Erjin Zhou, and Jian Sun. European Conference on Computer Vision (ECCV), 2020.
- [Rethinking on Multi-Stage Networks for Human Pose Estimation](https://arxiv.org/abs/1901.00148) Wenbo Li, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei and Jian Sun.
