# Big Sleep

此仓库包含MegEngine实现的多模态模型`Big Sleep`，其将`CLIP`与`BigGAN`的生成器相结合，用户可以轻松使用一行文本构想图像！

## 使用方法

请使用GPU设备，否则生成过程可能会过长。

使用`hub`加载

```python
from megengine import hub
modelhub = hub.import_module(repo_info='megengine/models', git_host='github.com')

dream = modelhub.Imagine(
    # 需要进行构想的文本
    text = "fire in the sky", 
    # 传入参考图像用于稍微引导生成
    img = None,
    # 生成图像尺寸大小
    image_size=512,
    # 迭代过程的学习率
    lr = 5e-2,
    # 保存图像的间隔
    save_every = 25,
    # 是否保存迭代过程中的所有图像，否则图像将会重写到一张图片上
    save_progress = True,
    # 惩罚关键词
    text_min = None,
    # 梯度累积的步数
    gradient_accumulate_every: int = 1,
    epochs: int = 20,
    iterations: int = 1050,
    # 是否将迭代过程中的所有图像保存为mp4视频文件
    animate: bool = False,
    # 保存mp4的帧率
    fps: int = 15,
    # BIgSleep中采样方式
    bilinear: bool = False,
    # 固定随机种子
    seed: Optional[int] = None,
    # 限制最大类别数量
    max_classes: Optional[int] = None,
    # 用于可微topk
    class_temperature: float = 2.,
    # 保存文件时是否加上日期前缀
    save_date_time: bool = False,
    # 是否保存得分最高的图像
    save_best: bool = True,
    # 实验性采样
    experimental_resample: bool = False,
    ema_decay: float = 0.99,
    num_cutouts: int = 128,
    center_bias: bool = False,
    clip_type: str = 'RN50',
    root: str = 'BigSleep',
)

# 开始迭代生成图像
dream()
```

本地加载

```python
from official.multimodal.big_sleep import Imagine

dream = Imagine(
    text = "fire in the sky",
    lr = 5e-2,
    save_every = 25,
    save_progress = True,
    image_size=512
)

# 开始迭代生成图像
dream()
```

### 参考

[lucidrains/big-sleep](https://github.com/lucidrains/big-sleep)
