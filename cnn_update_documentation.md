
# Text Classification with CNN

This repository contains a script for text classification using a Convolutional Neural Network (CNN) implemented with PyTorch. The script handles data preprocessing, model training, evaluation, and prediction on a dataset of Chinese text reviews.

## Dependencies

The following libraries are required to run the script:

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

## Data Loading and Preprocessing

### Loading the Dataset

The dataset is loaded from an Excel file and the necessary columns are selected and renamed.

```python
df = pd.read_excel('E://AISpeech/data/testdata_withood_2w_noentity.xlsx')
df = df[['领域', '用户query']]
df.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
```

### Data Cleaning

Remove rows with null values and clean the text by removing punctuation.

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

### Loading Stopwords

Load the list of stopwords from a file.

```python
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwords = stopwordslist('E://AISpeech/chineseStopWords.txt')
```

### Text Tokenization

Tokenize the text using Jieba and remove stopwords.

```python
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
```

### Text to Sequence

Convert the text into sequences for model input.

```python
def text_to_sequence(text, vocab, max_length):
    # Implementation details
df['sequence'] = df['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
```

## Dataset Splitting and Loading

### Split the Dataset

Split the dataset into training and testing sets.

```python
X = np.array(df['sequence'].tolist())
y = np.array(df['cat_id'].tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Custom Dataset Class

Define a custom PyTorch dataset class for loading the data.

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

### Data Loaders

Create data loaders for batch processing.

```python
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## Model Construction and Training

### Define the CNN Model

Define a simple CNN model using PyTorch.

```python
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # Model layers

    def forward(self, x):
        # Forward pass
```

### Train the Model

Define the training loop and use an optimizer and loss function for training the model.

```python
model = TextCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        # Training steps
```

## Model Evaluation and Prediction

### Evaluate the Model

Evaluate the model on the test set and print the classification report and confusion matrix.

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

### Save Prediction Results

Save the prediction results and probabilities to an Excel file.

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

## Handling OOD Data

### Process OOD Data

Preprocess and predict the categories for new OOD data.

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

### Save OOD Prediction Results

Save the OOD data prediction results to an Excel file.

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

This document provides a detailed explanation of the script, covering data loading, preprocessing, model training, evaluation, and prediction.
