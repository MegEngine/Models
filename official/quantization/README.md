模型量化 Model Quantization
---

本目录包含了采用MegEngine实现的量化训练和部署的代码，包括常用的ResNet、ShuffleNet和MobileNet，其量化模型的ImageNet Top 1 准确率如下：

| Model | top1 acc (float32) | FPS* (float32) | top1 acc (int8) | FPS* (int8) |
| --- | --- | --- | --- | --- |
| ResNet18 |  70.312  |    | 
| ResNet50 | | |
| ShufflenetV1 |   |   |
| MobilenetV2 |   |    |

**: FPS is measured on Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz*

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

# fake quant
Q.quantize_qat(model, qconfig=Q.ema_fakequant_qconfig)

# real quant
Q.quantize(model)

@jit.trace(symbolic=True):
def inference_func(x):
    return model(x)

inference_func.dump(...)
```