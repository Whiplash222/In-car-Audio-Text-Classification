# Benz-Text-Classification
## 背景和需求
希望实现自动化文本分类算法模型，对奔驰人机对话中的未落域文本数据，进行自动化标注分类。未落域数据的目标分类类别标签如下：

1.语气词
2.无实体/截断/停顿
3.控单实体/完整句
4.媒体单实体/完整句
5.地理位置实体/完整句
6.其他实体
7.误唤醒
8.EC
9.纯数字
10.骂傻
11.疑似唤醒
12.无明确意图
13.正常退出
14.科普
15.电话
<table>
  <tr>
    <td>分类</td>
    <td>描述</td>
    <td>示例</td>
  </tr>
  <tr>
    <td>1.语气词</td>
    <td>- 口语中常见但无实际含义 
    - 识别模型未做文本顺滑，可能导致语气词较多而影响NLU效果</td>
    <td>啊；嗯；a；哈哈哈；</td>
  </tr>
</table>

## 分类方法与代码说明
基于未落域数据目标分类各类的不同特点，以及未落域与已落域数据的共性与差异性，我们采取规则判断与深度学习相结合的方法对未落域文本数据进行分类，具体分类流程如下图所示：

![Image](https://github.com/Whiplash222/Benz-Text-Classification/blob/main/AISpeech_update.png)

也即，我们对*3.控单实体/完整句 4.媒体单实体/完整句*这两类在已落域文本中有可用训练数据，且类别规则判断较为复杂的类别进行深度学习训练分类，而对其他具有较为清晰简单定义的类别进行规则判断分类。

### CNN分类训练具体步骤与代码说明
这是一个使用 PyTorch 实现卷积神经网络 (CNN) 进行文本分类的代码说明，该代码包括数据预处理、模型训练、模型评估和未落域文本数据分类四部分。

#### 相关库加载

首先运行该脚本需要加载以下库：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import jieba as jb
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
```

#### 数据加载和预处理

##### 加载数据集

首先从Excel文件加载训练数据集，本案例中，在多种尝试后，我们选择[training data](https://github.com/Whiplash222/Benz-Text-Classification/blob/main/testdata_noood_2w_noentity.xlsx)

```python
df = pd.read_excel('E://AISpeech/data/testdata_withood_2w_noentity.xlsx')
df = df[['领域', '用户query']]
df.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
```

### 数据清洗

移除空值并通过去除标点符号来清理文本。

```python
df = df[pd.notnull(df['review'])]

def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line
```

### 加载停用词

从文件中加载停用词列表。

```python
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('E://AISpeech/chineseStopWords.txt')
```

### 文本切词

使用结巴分词对文本进行切词，并去除停用词。

```python
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
```

### 文本序列化

将文本转换为序列以便输入到模型中。

```python
def text_to_sequence(text, vocab, max_length):
    # 实现细节
df['sequence'] = df['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
```




















