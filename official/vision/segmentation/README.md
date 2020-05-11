# Semantic Segmentation

本目录包含了采用MegEngine实现的经典[Deeplabv3plus](https://arxiv.org/abs/1802.02611.pdf)网络结构，同时提供了在PASCAL VOC和Cityscapes数据集上的完整训练和测试代码。

网络在PASCAL VOC2012验证集的性能和结果如下:

 Methods       | Backbone    | TrainSet  | EvalSet | mIoU_single   | mIoU_multi  |
 :--:          |:--:         |:--:       |:--:     |:--:           |:--:         |
 Deeplabv3plus | Resnet101   | train_aug | val     | 79.0          | 79.8        |


## 参考文献

- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611.pdf), Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and
Hartwig Adam; ECCV, 2018
