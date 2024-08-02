# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:15:25 2024

@author: YUE
"""

def main():
    
    #coding=utf-8
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
    import os
    
    
    
    # Load dataset
    df = pd.read_excel('./testdata_withood_2w_noentity.xlsx')
    df = df[['领域', '用户query']]
    df.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
    print("数据总量: %d ." % len(df))
    
    # Data cleaning
    df0 = df[pd.notnull(df['review'])]
    
    df_music=df0
    df_vehicle = df0[df0['cat'] != 'OOD']
    
    for iter in range(2):
        if iter==0:
            df=df_music
        else:
            df=df_vehicle
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
        
        # Load stopwords
        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
            return stopwords
        
        stopwords = stopwordslist('./chineseStopWords.txt')
        
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
        
        # Create PyTorch Dataset
        class TextDataset(Dataset):
            def __init__(self, X, Y):
                self.X = torch.tensor(X, dtype=torch.long)
                self.Y = torch.tensor(Y, dtype=torch.long)
        
            def __len__(self):
                return len(self.X)
        
            def __getitem__(self, idx):
                return self.X[idx], self.Y[idx]
            
        # Create PyTorch Dataset
        class TextDataset_ood(Dataset):
            def __init__(self, X):
                self.X = torch.tensor(X, dtype=torch.long)
        
            def __len__(self):
                return len(self.X)
        
            def __getitem__(self, idx):
                return self.X[idx]
        
        X = np.array(df['sequence'].tolist())
        Y = df['cat_id'].values
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
        
        train_dataset = TextDataset(X_train, Y_train)
        test_dataset = TextDataset(X_test, Y_test)
        
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
        
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
        # # 将数据转换为 DataFrame
        # data = {
        #     'y_true': y_true,
        #     'y_pred': y_pred
        # }
        
        # # 将probabilities添加到DataFrame中
        # for i in range(len(probabilities[0])):  # 假设每个概率向量的长度相同
        #     data[f'prob_class_{i}'] = [prob[i].item() for prob in probabilities]
        
        
        # df = pd.DataFrame(data)
        
        # # 输出为Excel文件
        # output_file = './predictions.xlsx'
        # df.to_excel(output_file, index=False, engine='openpyxl')
        
        # print(f'Data saved to {output_file}')
        
        #ood data
        ooddata = pd.read_excel('./top1000new.xlsx')
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
        
        # 将数据转换为 DataFrame
        data = {
            'y_pred': y_pred_ood,
            'sequence': x_input_ood
        }
        
        # 将probabilities添加到DataFrame中
        for i in range(len(probabilities_ood[0])):  # 假设每个概率向量的长度相同
            data[f'prob_class_{i}'] = [prob[i].item() for prob in probabilities_ood]
        if iter==0:
            data = pd.DataFrame(data)
            data['maxprob'] = data[['prob_class_0', 'prob_class_1', 'prob_class_2']].max(axis=1)
            
            
            # 输出为Excel文件
            output_file = './predictions_music_.xlsx'
            data.to_excel(output_file, index=False, engine='openpyxl')
            
            print(f'Music Data saved to {output_file}')
            cat_id_music=cat_id_df
            data_music=data
        else:
            data = pd.DataFrame(data)
            data['maxprob'] = data[['prob_class_0', 'prob_class_1']].max(axis=1)
            
            
            # 输出为Excel文件
            output_file = './predictions_vehicle_.xlsx'
            data.to_excel(output_file, index=False, engine='openpyxl')
            
            print(f'Vehicle Data saved to {output_file}')
            cat_id_vehicle=cat_id_df
            data_vehicle=data
    
if __name__ == "__main__":
    main()
    
