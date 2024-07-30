
# 使用 CNN 进行文本分类

此存储库包含一个使用 PyTorch 实现卷积神经网络 (CNN) 进行文本分类的脚本。该脚本处理数据预处理、模型训练、评估和数据集中文文本评论的预测。

## 依赖库

运行此脚本需要以下库：

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

## 数据加载和预处理

### 加载数据集

从Excel文件加载数据集，选择必要的列并重命名。

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

## 数据集划分和加载

### 划分数据集

将数据集划分为训练集和测试集。

```python
X = np.array(df['sequence'].tolist())
y = np.array(df['cat_id'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 自定义数据集类

定义一个自定义的 PyTorch 数据集类以加载数据。

```python
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)
```

### 数据加载器

使用 `DataLoader` 创建数据加载器以进行批处理。

```python
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## 模型构建和训练

### 定义 CNN 模型

使用 PyTorch 定义一个简单的 CNN 模型。

```python
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # 模型层

    def forward(self, x):
        # 前向传播
```

### 训练模型

定义训练循环并使用优化器和损失函数来训练模型。

```python
model = TextCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        # 训练步骤
```

## 模型评估和预测

### 评估模型

在测试集上评估模型性能并打印分类报告和混淆矩阵。

```python
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        y_pred.extend(outputs.argmax(1).tolist())
        y_true.extend(labels.tolist())
print(classification_report(y_true, y_pred, target_names=cat_id_df['cat'].values))
```

### 保存预测结果

将预测结果和概率保存到 Excel 文件中。

```python
data = {
    'y_true': y_true,
    'y_pred': y_pred
}
for i in range(len(probabilities[0])):
    data[f'prob_class_{i}'] = [prob[i].item() for prob in probabilities]
df = pd.DataFrame(data)
output_file = 'E://predictions.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
```

## 处理 OOD 数据

### 处理 OOD 数据

对新的 OOD 数据进行相同的预处理并预测其类别。

```python
ooddata = pd.read_excel('E://AISpeech/data/top1000new.xlsx')
ooddata = ooddata[['领域', '用户query']]
ooddata.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
ooddata = ooddata[pd.notnull(ooddata['review'])]
ooddata['clean_review'] = ooddata['review'].apply(remove_punctuation)
ooddata['cut_review'] = ooddata['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
ooddata['sequence'] = ooddata['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
X_ood = np.array(ooddata['sequence'].tolist())
ood_dataset = TextDataset(X_ood)
ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False)
```

### 保存 OOD 预测结果

将 OOD 数据的预测结果保存到 Excel 文件中。

```python
data = {
    'y_pred': y_pred_ood,
    'sequence': x_input_ood
}
for i in range(len(probabilities_ood[0])):
    data[f'prob_class_{i}'] = [prob[i].item() for prob in probabilities_ood]
df = pd.DataFrame(data)
output_file = 'E://predictions_withood_noentity.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
```

本文档提供了脚本的详细说明，涵盖了数据加载、预处理、模型训练、评估和预测的全过程。
