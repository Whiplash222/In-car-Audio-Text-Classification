# In-car Audio-Text-Classification
## 目录
- [背景和需求](#背景和需求)
- [分类方法与代码说明](#分类方法与代码说明)
  - [CNN分类训练具体步骤与代码说明](#CNN分类训练具体步骤与代码说明)
  - [规则分类具体步骤与代码说明](#规则分类具体步骤与代码说明)
  - [结果整理与展示](#结果整理与展示)
  - [其他方法探索](#其他方法探索)
   
## 背景和需求
希望实现自动化文本分类算法模型，对汽车语音人机对话中的未落域文本数据，进行自动化标注分类。未落域数据的目标分类类别标签如下：

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

## 分类方法与代码说明
基于未落域数据目标分类各类的不同特点，以及未落域与已落域数据的共性与差异性，我们采取规则判断与深度学习相结合的方法对未落域文本数据进行分类，具体分类流程如下图所示：

![Image](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/framework.png)

也即，我们对*3.控单实体/完整句 4.媒体单实体/完整句*这两类在已落域文本中有可用训练数据，且类别规则判断较为复杂的类别进行深度学习训练分类，而对其他具有较为清晰简单定义的类别进行规则判断分类。

### CNN分类训练具体步骤与代码说明
这是一个使用 PyTorch 实现卷积神经网络 (CNN) 进行文本分类的代码说明，该代码包括数据预处理、模型训练、模型评估和未落域文本数据分类四部分。具体代码详见【[cnn_update.py](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/cnn_update.py)】

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

首先从Excel文件加载训练数据集，本案例中，在多种尝试后，我们选择2万条媒体单实体/完整句以及2万条控单实体/完整句构成的训练数据集，来进行*媒体单实体/完整句*类别的分类训练，而利用2万条媒体单实体/完整句、2万条控单实体/完整句以及2万条未落域数据构成的训练数据集，来进行*控单实体/完整句*类别的分类训练。可通过更改读取输入的训练数据集来实现其他文本分类的运行实现。

```python
df = pd.read_excel('/testdata_withood_2w_noentity.xlsx')
df = df[['领域', '用户query']]
df.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
```

##### 数据清洗

通过移除空值并去除标点符号来对文本数据进行清理：

```python
df = df[pd.notnull(df['review'])]

# Encode categories
label_encoder = LabelEncoder()
df['cat_id'] = label_encoder.fit_transform(df['cat'])
cat_id_df = pd.DataFrame({'cat': label_encoder.classes_, 'cat_id': range(len(label_encoder.classes_))})

# Text cleaning function
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line
```

##### 加载停用词

从[停用词列表](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/chineseStopWords.txt)中加载中文停用词列表，使用jieba对文本进行分词，并去除停用词。

```python
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('/chineseStopWords.txt')

df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
```

##### 文本序列化

将文本转换为序列以便输入到模型中：

```python
# Apply text cleaning
df['clean_review'] = df['review'].apply(remove_punctuation)
# Tokenization and padding
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
# Build vocabulary
vocab = {}
for text in df['cut_review']:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab) + 1

MAX_SEQUENCE_LENGTH = 10

# Convert text to sequences
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()]
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence
    else:
        sequence = sequence[:max_len]
    return sequence

df['sequence'] = df['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
```

#### 数据集划分和加载

将数据集划分为训练集和测试集：

```python
X = np.array(df['sequence'].tolist())
y = np.array(df['cat_id'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

定义一个自定义的 PyTorch 数据集类以加载数据：

```python
# Create PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
```

使用 `DataLoader` 创建数据加载器以进行批处理：

```python
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

#### 模型构建和训练

##### 定义 CNN 模型

使用 PyTorch 定义构建 CNN 模型：

```python
# Define the CNN model
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_size, kernel_sizes=[3, 4, 5], num_filters=100, drop_prob=0.5):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)
```

##### 训练模型

对模型参数进行初始化，确定优化器和损失函数，定义训练循环：

```python
# Initialize the model, loss function, and optimizer
vocab_size = len(vocab) + 1
embedding_dim = 100
output_size = len(cat_id_df)
kernel_sizes = [3, 4, 5]
num_filters = 100

model = CNNClassifier(vocab_size, embedding_dim, output_size, kernel_sizes, num_filters)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 20
model.train()

history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)

    val_loss = 0
    val_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
    model.train()

    history['train_loss'].append(epoch_loss / len(train_loader))
    history['train_accuracy'].append(epoch_accuracy / len(train_loader))
    history['val_loss'].append(val_loss / len(test_loader))
    history['val_accuracy'].append(val_accuracy / len(test_loader))

    print(f'Epoch {epoch + 1}/{epochs}, '
          f'Train Loss: {epoch_loss / len(train_loader)}, Train Accuracy: {epoch_accuracy / len(train_loader)}, '
          f'Val Loss: {val_loss / len(test_loader)}, Val Accuracy: {val_accuracy / len(test_loader)}')
```

#### 模型评估和预测

##### 评估模型

在此前划分出来的测试集上评估模型性能并输出分类报告和混淆矩阵：

```python
# Evaluating the model
model.eval()
test_loss = 0
test_accuracy = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0)

print(f'Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {test_accuracy / len(test_loader)}')

# Plotting the training history (loss and accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title('Loss')
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='test')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.plot(history['train_accuracy'], label='train')
plt.plot(history['val_accuracy'], label='test')
plt.legend()
plt.show()

# Confusion Matrix
y_pred = []
y_true = []
probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probabilities.extend (torch.softmax(outputs, dim=1))
        y_pred.extend(outputs.argmax(1).tolist())
        y_true.extend(labels.tolist())

conf_mat = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=cat_id_df['cat'].values, yticklabels=cat_id_df['cat'].values)
plt.ylabel('real label', fontsize=18)
plt.xlabel('predicted label', fontsize=18)
plt.show()

print('accuracy %s' % accuracy_score(y_pred, y_true))
print(classification_report(y_true, y_pred, target_names=cat_id_df['cat'].values))
print(probabilities)
```

#### OOD数据分类

##### 预处理OOD数据

对出现频率处于前1000的[OOD数据](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/top10.xlsx)(此处由于信息保密的原因我们仅展示前10条OOD数据)进行与上述训练数据相通的预处理并利用上面训练好的CNN分类器对新到达的OOD数据进行文本分类预测：

```python
#ood data
ooddata = pd.read_excel('/top1000new.xlsx')
ooddata = ooddata[['领域', '用户query']]
ooddata.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
ooddata = ooddata[pd.notnull(ooddata['review'])]
ooddata['clean_review']= ooddata['review'].apply(remove_punctuation)
ooddata['cut_review'] = ooddata['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
ooddata['sequence'] = ooddata['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
X_ood=np.array(ooddata['sequence'].tolist())
ood_dataset=TextDataset_ood(X_ood)
ood_loader=DataLoader(ood_dataset, batch_size=64, shuffle=False)

# Confusion Matrix
y_pred_ood = []
probabilities_ood = []
x_input_ood = []

with torch.no_grad():
    for inputs in ood_loader:
        outputs = model(inputs)
        probabilities_ood.extend(torch.softmax(outputs, dim=1))
        y_pred_ood.extend(outputs.argmax(1).tolist())
        x_input_ood.extend(inputs.tolist())
```

##### 输出OOD预测结果

将OOD数据的预测分类结果以及属于每一类的概率值输出为excel文件：
```python
data = {
    'y_pred': y_pred_ood,
    'sequence': x_input_ood
}
for i in range(len(probabilities_ood[0])):
    data[f'prob_class_{i}'] = [prob[i].item() for prob in probabilities_ood]
df = pd.DataFrame(data)
output_file = '/predictions_withood_noentity.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')
```

### 规则分类具体步骤与代码说明
对除了*3.控单实体/完整句 4.媒体单实体/完整句*这两类外，其他具有较为清晰简单定义的类别，我们采用规则判断的方式进行文本分类。具体代码详见【[rulecheck.py](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/rulecheck.py)】
（可以通过增添减少或修改规则判断的类别所需用到的判定关键词文件的方式，对规则分类操作进行调整修改。）

### 结果整理与展示
对于规则判断后仍未落域的数据，根据算法训练得到的属于每一类的概率值，人为判断设定一个阈值，当落域概率值大于这一阈值时，将这一文本数据归为对应的类别，其他未落域数据全部标为“无明确意图”。文件[result](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/rule_plus_cnn_result.xlsx)中计算并展示了整体与各类别对应的精确率和召回率，可以看到整体分类精确率达到了87.60%，说明利用规则判断+深度学习进行文本分类的方法，能够对该未落域数据实现较好的分类结果。

### 其他方法探索
我们还尝试了无监督文本分类方法潜在狄利克雷分配 Latent Dirichlet Allocation(LDA)，方法大概思路如下：

![image](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/LDA_method.png)

LDA方法实现代码详见：[read_lda_summary.py](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/read_lda_summary.py)

未落域数据的准确率与召回率结果见Excel文件：[result_lda](https://github.com/Whiplash222/In-car-Audio-Text-Classification/blob/main/OOD%20data_lda%20results.xlsx).

可以看到由于未落域数据类别过多区分困难，无监督的文本分类方法无法较好地对未落域数据进行分类，结果远远不如规则判断+深度学习的结果。

后续可继续探索针对训练数据不足或者训练数据与待分类数据类别有所区别的情况下，迁移学习、zero-shot/few-shot learning的方法，并与已有方法进行对比尝试。








