模型量化 Model Quantization
---

本目录包含了采用MegEngine实现的检测模型量化训练和部署的代码，包括常用的RetinaNet，因为时间原因，仅仅量化了检测模型的Resnet-Backbone部分，后续会随着MegEngine的版本持续更新。

| Model | mAP | test speed* (float32) | mAP (int8) | test speed* (int8) |
| --- | --- | --- | --- | --- |
| RetinaNet |  29.9  | 42it/s(8GPU)   | 29.7 | 45it/s(8GPU) |

**: FPS is measured on Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz, standard COCO input format*

*We finetune mobile models with QAT for 3 epochs, training longer may yield better accuracy*

量化模型使用时，统一读取0-255的uint8图片，减去128的均值，转化为int8，输入网络。


#### (Optional) Download Pretrained Models for Detection backbone
```
wget https://data.megengine.org.cn/models/weights/resnet18_normal_69824.pkl
```

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

# HOWTO use this codebase

## Step 1. Train a fp32 model

```
python3 train.py -f retinanet_res18_coco_1x_800size.py -w path/to/resnet18_normal_69824.pkl
```

## Step 2. Finetune fp32 model with quantization aware training(QAT)

```
python3 finetune.py -f retinanet_res18_coco_1x_800size_finetune.py -w path/to/retinanet_weights/ckpt.pkl
```

## Step 3. Test QAT model on COCO Testset

```
python3 test.py -f retinanet_res18_coco_1x_800size_finetune.py -n 8 -m path/to/finetuned_weights/ckpt.pkl --mode qat
```

or testing in quantized mode, which uses only cpu for inference and takes longer time

```
python3 test.py -f retinanet_res18_coco_1x_800size_finetune.py -n 1 -m path/to/finetuned_weights/ckpt.pkl --mode quantized
```
