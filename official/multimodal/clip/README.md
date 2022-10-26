# CLIP

此仓库包含MegEngine实现的多模态模型`CLIP`，但不包含训练及测试代码。

`models.py`中实现了CLIP的不同配置：`RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`, `ViT-B-32`, `ViT-B-16`, `ViT-L-14`和`ViT-L-14-336px`。

在ImageNet V2 matched-frequency数据集上，以float16的精度达成了一下的零样本分类准确度

| 模型           | TOP-1  |TOP-5  |
| -------------- | -------|------|
| RN50           | 53.55% |81.53%|
| RN101          | 56.21% |83.77%|
| RN50x4         | 59.77% |85.90%|
| RN50x16        | 64.14% |88.39%|
| RN50x64        | 66.90% |90.46%|
| ViT-B-32       | 56.48% |83.57%|
| ViT-B-16       | 62.24% |87.72%|
| ViT-L-14       | 69.72% |90.89%|
| ViT-L-14-336px | 70.72% |91.68%|

## 零样本（zero-shot）分类

用户可以使用以下模板使用`CLIP`进行零样本图像分类。

### 加载网络

```python
import megengine as mge
from megengine import hub
modelhub = hub.import_module(repo_info='megengine/models', git_host='github.com')

# 加载网络结构及预训练模型
# 方式一
clip = hub.load("megengine/models", "rn50", pretrained=True)
clip.eval()

# 将网络部分权重转换为float16, 仅限GPU
clip.convert_weights('float16')

# 方式二
# 查看所有可用模型
print(CLIP.available_models())

# 直接使用 from_pretrained 方法加载模型即可
clip = CLIP.from_pretrained(model_name='RN50', dtype='float16')

# 查看网络配置信息
clip.model_config()

# 使用float32的精度推理
clip.convert_weigths('float32')
```

### 数据处理

```python
import cv2
from megengine.data.transform import CenterCrop, Compose, Normalize, Resize

#数据处理
image_resolution = clip.image_resolution  # clip需要固定输入图片的大小
transfroms =  Compose([
    Resize(image_resolution, interpolation=cv2.INTER_CUBIC),
    CenterCrop(image_resolution),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

```

数据处理构建完毕后需要用户手动构建`Dataloader`。

### 构建文本模板和类别

`CLIP`需要一些文本模板/提示来描述某一张图片，比如：`a photo of {}.`，`a photo of many {}.`等，大括号中可以填入各种类别名称。这样为每一个类别都生成n句话，再使用文本编码器和图片编码器的输出向量做相似度计算，得分高者则认为其为该类的概率更高。

`CLIP`中内置了imagenet的80个文本模板，这里使用内置的CLIP推理工具，使用方法如下。

```python
utils = modelhub.ClipInferenceUtils
```

随后调用如下方法即可得到对应的文本模板。

```python
imagenet_templates = utils.generate_imagenet_templates()
```

对于不同的数据集可以采用不同的文本模板，其格式如下：

```python
templates: List[str] = [
 'a bad photo of a {}.',
 'a photo of many {}.',
 ...
]
```

同时我们需要各个类别的名称，可通过调用以下代码得到imagenet的1000个类别。

```python
imagenet_classes = utils.generate_imagenet_classes()
```

对于不同的数据集需要使用对应的类别名称，其格式如下：

```python
classes：List[str] = [
    'tench',
    'goldfish',
    ...
]
```

### 生成零样本分类权重

使用下列代码生成权重。

```python
zeroshot_wieghts = utils.generate_zeroshot_classifier_weight(clip, imagenet_classes, imagenet_templates)
```

### 预测

传入模型、dataloader和零样本权重即可进行预测

```python
top1, top5 = utils.predict(clip, loader, zeroshot_wieghts, logit_scale=100.)
print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")
```

如果你只想预测一张图片，使用`predict_once`方法即可

```python
logits = utils.predict_once(clip, image, zeroshot_wieghts, logit_scale=100.)
```

## 参考

[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

[openai/CLIP](https://github.com/openai/CLIP)
