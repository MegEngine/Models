# MegEngine classification models

图像分类是计算机视觉的基础任务。许多计算机视觉的其它任务（例如物体检测）都使用了基于图像分类的预训练模型。因此，我们提供了各种在ImageNet上预训练好的分类模型，
包括[ResNet](./resnet)系列, [Shufflenet](./shufflenet)系列等，这些模型在**ImageNet验证集**上的测试结果如下表：

| 模型 | top1 acc | top5 acc |
| --- | :---: | :---: |
| ResNet18 | 70.312 | 89.430 |
| ResNet34 | 73.960 | 91.630 |
| ResNet50 | 76.254 | 93.056 |
| ResNet101 | 77.944 | 93.844 |
| ResNet152 | 78.582 | 94.130 |
| ResNeXt50 32x4d | 77.592 | 93.644 |
| ResNeXt101 32x8d| 79.520 | 94.586 |
| ShuffleNetV2 x0.5 | 60.696 | 82.190 |
| ShuffleNetV2 x1.0 | 69.372 | 88.764 |
| ShuffleNetV2 x1.5 | 72.806 | 90.792 |
| ShuffleNetV2 x2.0 | 75.074 | 92.278 |
