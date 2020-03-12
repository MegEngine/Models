# BERT

此仓库包含MegEngine实现的经典`BERT`网络结构，还提供了有关GLUE MRPC任务的完整培训和测试代码。

你可以调用以下预训练模型, 在不同的下游任务中进行finetune.

| 模型                       | 字典 | 配置 |
| ---                        |  --- |  --- |
| wwm_cased_L-24_H-1024_A-16|   [link](https://data.megengine.org.cn/models/weights/bert/wwm_cased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/wwm_cased_L-24_H-1024_A-16/bert_config.json)
| wwm_uncased_L-24_H-1024_A-16|   [link](https://data.megengine.org.cn/models/weights/bert/wwm_uncased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/wwm_uncased_L-24_H-1024_A-16/bert_config.json)
| cased_L-12_H-768_A-12|   [link](https://data.megengine.org.cn/models/weights/bert/cased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/cased_L-12_H-768_A-12/bert_config.json)
| cased_L-24_H-1024_A-16|   [link](https://data.megengine.org.cn/models/weights/bert/cased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/cased_L-24_H-1024_A-16/bert_config.json)
| uncased_L-12_H-768_A-12|   [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-12_H-768_A-12/bert_config.json)
| uncased_L-24_H-1024_A-16|   [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-24_H-1024_A-16/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/uncased_L-24_H-1024_A-16/bert_config.json)
| chinese_L-12_H-768_A-12|   [link](https://data.megengine.org.cn/models/weights/bert/chinese_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/chinese_L-12_H-768_A-12/bert_config.json)
| multi_cased_L-12_H-768_A-12|   [link](https://data.megengine.org.cn/models/weights/bert/multi_cased_L-12_H-768_A-12/vocab.txt) | [link](https://data.megengine.org.cn/models/weights/bert/multi_cased_L-12_H-768_A-12/bert_config.json)


模型的权重来自Google的pre-trained models, 其含义也与其一致, 用户可以直接使用`megengine.hub`轻松的调用预训练的bert模型, 以及下载对应的`vocab.txt`与`bert_config.json`. 我们在[models](./official/nlp/bert)中还提供了更加方便的脚本, 可以通过任务名直接获取到对应字典, 配置, 与预训练模型.

## Training Example

```bash
python3 train.py \
  --do_lower_case \
  --max_seq_length 128 \
  --data_dir ./glue_data/MRPC \
  --pretrained_bert uncased_L-12_H-768_A-12 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
```

## Eval Example

```bash
python3 test.py \
  --do_lower_case \
  --max_seq_length 128 \
  --data_dir ./glue_data/MRPC \
  --pretrained_bert uncased_L-12_H-768_A-12 \
```

# How to Use
`model.py`, 用MegEngine实现的BERT模型, 已经相关的预训练模型设置

`mrpc_dataset.py`, 定义一个dataloader生成器，它可以生成处理过的MRPC数据，这些数据可以直接用于训练/评估。

`train.py`, 训练脚本

`test.py`, 测试脚本

`config.py`, 定义了所有的测试/训练需要的变量

- `--data_dir`, 输入数据目录。 该任务应包含.tsv文件（或其他数据文件）.(示例代码中支持MRPC数据集)
- `--max_seq_length`,  WordPiece tokenization之后的最大总输入序列长度。 长度大于此长度的序列将被截断，小于长度的序列将被填充。
- `--do_lower_case`, 如果使用的是无大小写的模型，请设置此标志。
- `--pretrained_bert`, 使用的pretrained bert, 例如`uncased_L-12_H-768_A-12`

**如果要运行训练脚本 `train.py`，则需要设置:** 

- `--train_batch_size`, 训练使用的batch_size, 默认`16`.
- `--eval_batch_size`, 测试使用batch_size, 默认`16`.
- `--learning_rate`, Adam的初始化learning rate, 默认`5e-5`.
- `--num_train_epochs`, 训练的总轮数, 默认`3`.
- `--save_model_path`, 需要save的模型路径, 默认`./check_point_last.pkl`.

**如果要运行训练脚本 `test.py`，则需要设置** 

- `--eval_batch_size`, 测试使用batch_size, 默认`16`.
- `--load_model_path`, 需要load的模型路径, 默认`./check_point_last.pkl`.

# Other Data Files

在运行此示例之前，您应该准备所有GLUE MRPC数据，您可以自己下载它或使用我们存储库中的备份。

`glue_data/MRPC`, MRPC原始数据的目录

# Results

在glue_data/MRPC数据集中使用默认的超参数进行微调和评估，评估结果介于84％和88％之间。

| Dataset | pretrained_bert | acc |
| --- |   --- |  --- |
| glue_data/MRPC |   uncased_L-12_H-768_A-12 |  86.25% |

# Reference project
- "PyTorch Pretrained Bert" <https://github.com/Meelfy/pytorch_pretrained_BERT>
