模型量化 Model Quantization
---

本目录包含了采用MegEngine实现的量化训练和部署的代码，包括常用的ResNet、ShuffleNet和MobileNet，其量化模型的ImageNet Top 1 准确率如下：

| Model | top1 acc (float32) | FPS* (float32) | top1 acc (int8) | FPS* (int8) |
| --- | --- | --- | --- | --- |
| ResNet18 |  69.824  | 10.5   | 69.754 | 16.3 |
| ShufflenetV1 (1.5x) | 71.954  |  17.3 | | 25.3 |
| MobilenetV2 | 72.820  |  13.1  |  | 17.4 |

**: FPS is measured on Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, single 224x224 image*

量化模型使用时，统一读取0-255的uint8图片，减去128的均值，转化为int8，输入网络。

# Brief Introduction to MegEngine.quantization

## Quantization Aware Training (QAT)

```python
import megengine.quantization as Q

model = ...

# Quantization Aware Training
Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

for _ in range(...):
    train(model)
```

## Deploying Quantized Model

```python
import megengine.quantization as Q
import megengine.jit as jit

model = ...

Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

# real quant
Q.quantize(model)

@jit.trace(symbolic=True):
def inference_func(x):
    return model(x)

inference_func.dump(...)
```

# HOWTO

## Step 1. Train a fp32 model

```
python3 train.py -a resnet18 -d /path/to/imagenet --mode normal
```

## Step 2. Finetune fp32 model with quantization aware training(QAT)

```
python3 finetune.py -a resnet18 -d /path/to/imagenet --checkpoint /path/to/resnet18.normal/checkpoint.pkl --mode qat
```

## Step 3. Test QAT model on ImageNet Testset

```
python3 test.py -a resnet18 -d /path/to/imagenet --checkpoint /path/to/resnet18.qat/checkpoint.pkl --mode qat
```

or testing in quantized mode, which uses only cpu for inference and takes longer time

```
python3 test.py -a resnet18 -d /path/to/imagenet --checkpoint /path/to/resnet18.qat/checkpoint.pkl --mode quantized -n 1
```

## Step 4. Inference and dump

```
python3 inference.py -a resnet18 --checkpoint /path/to/resnet18.qat/checkpoint.pkl --mode quantized --dump
```

will feed a cat image to the network and output the classification probabilities with quantized network.

Also, set `--dump` will dump the quantized network to `resnet18.quantized.megengine` binary file.

