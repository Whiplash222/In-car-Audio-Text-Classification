# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:22:29 2024

@author: YUE
"""


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
import jieba.posseg as pseg
import spacy
import re

# 加载中文语言模型
nlp = spacy.load('zh_core_web_sm')

# Encode categories
label_encoder = LabelEncoder()

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

stopwords = stopwordslist('E://AISpeech/chineseStopWords.txt')



MAX_SEQUENCE_LENGTH = 10

# Convert text to sequences
def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 0) for word in text.split()]
    if len(sequence) < max_len:
        sequence = [0] * (max_len - len(sequence)) + sequence
    else:
        sequence = sequence[:max_len]
    return sequence

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

#ood data
ooddata = pd.read_excel('E://AISpeech/data/top1000new.xlsx')
ooddata = ooddata[['领域', '用户query']]
ooddata.rename(columns={'领域': 'cat', '用户query': 'review'}, inplace=True)
n=len(ooddata['review'])
# ooddata = ooddata[pd.notnull(ooddata['review'])]
# ooddata['clean_review']= ooddata['review'].apply(remove_punctuation)
# ooddata['cut_review'] = ooddata['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
# ooddata['sequence'] = ooddata['cut_review'].apply(lambda x: text_to_sequence(x, vocab, MAX_SEQUENCE_LENGTH))
# X_ood=np.array(ooddata['sequence'].tolist())
# ood_dataset=TextDataset_ood(X_ood)
# ood_loader=DataLoader(ood_dataset, batch_size=64, shuffle=False)


# # 输出为Excel文件
# output_file = 'E://predictions_cnnnew.xlsx'
# df.to_excel(output_file, index=False, engine='openpyxl')

# print(f'Data saved to {output_file}')

# # # 纯数字
def is_pure_number(text):
    """
    判断文本是否是纯数字
    :param text: 输入的文本
    :return: 如果文本是纯数字则返回True，否则返回False
    """
    return text.isdigit()

# # # # 语气词
# def is_intonation_word(text, intonation_words):
#     """
#     判断文本是否完全是一个语气词
#     :param text: 输入的文本
#     :param intonation_words: 语气词关键词列表
#     :return: 如果文本完全是一个语气词则返回True，否则返回False
#     """
#     return text in intonation_words
df_intonation = pd.read_excel("E://AISpeech/data/intonation_phrases_10000.xlsx")
df_intonation = df_intonation.drop_duplicates()
intonation_words = df_intonation['Intonation Words'].tolist()

def is_composed_of_keywords(word, keywords_set):
    return all(char in keywords_set for char in word)

# # # 误唤醒
def contains_miswake_word(text, miswake_words):
    """
    判断文本是否是误唤醒
    """
    for word in miswake_words:
        if word in text:
            return True
    return False
df_miswake = pd.read_excel("E://AISpeech/data/miswake_phrases.xlsx")
df_miswake = df_miswake.drop_duplicates()
miswake_words = df_miswake['Miswake Words'].tolist()

# # # 骂傻
def contains_insult_word(text, insult_words):
    """
    判断文本是否是骂傻
    """
    for word in insult_words:
        if word=='靠':
            if text=='靠':
                return True
        else:
            if word in text:
                return True
    return False
df_insult = pd.read_excel("E://AISpeech/data/insult_phrases.xlsx")
df_insult = df_insult.drop_duplicates()
insult_words = df_insult['Insult Words'].tolist()

# # # 电话
def contains_phone_word(text, phone_words):
    """
    判断文本是否是电话
    """
    for word in phone_words:
        if word in text:
            return True
    return False
phone_words=['号码','电话']

# # # 疑似唤醒
def contains_wake_word(text, wake_words):
    """
    判断文本是否是疑似唤醒
    """
    for word in wake_words:
        if word in text:
            return True
    return False
df_wake = pd.read_excel("E://AISpeech/data/wake_phrases.xlsx")
df_wake = df_wake.drop_duplicates()
wake_words = df_wake['Wake Words'].tolist()

# # # 正常退出
def contains_exit_word(text, exit_words):
    """
    判断文本是否是正常退出
    """
    for word in exit_words:
        if word in text:
            return True
    return False
df_exit = pd.read_excel("E://AISpeech/data/exit_keywords.xlsx")
df_exit = df_exit.drop_duplicates()
exit_words = df_exit['Exit Keywords'].tolist()

# # # 科普
def contains_science_word(text, science_words):
    """
    判断文本是否是科普
    """
    for word in science_words:
        if word in text:
            return True
    return False
df_science = pd.read_excel("E://AISpeech/data/science_keywords.xlsx")
df_science = df_science.drop_duplicates()
science_words = df_science['Science Keywords'].tolist()

# # # EC
def contains_EC_word(text, EC_words):
    """
    判断文本是否是EC
    """
    for word in EC_words:
        if word in text:
            return True
    return False
df_EC = pd.read_excel("E://AISpeech/data/navigation_key_phrases.xlsx")
df_EC = df_EC.drop_duplicates()
EC_words = df_EC['Navigation_Key_Phrase'].tolist()

# # # 无实体
def contains_noentity_word(text, noentity_words):
    """
    判断文本是否是无实体
    """
    for word in noentity_words:
        if (word in text) & (len(text)<=4):
            return True
    return False
df_noentity = pd.read_excel("E://AISpeech/data/noentity_phrases.xlsx")
df_noentity = df_noentity.drop_duplicates()
noentity_words = df_noentity['Noentity_Key_Phrase'].tolist()

# # # 地理位置信息
# def contains_geographic_location(phrase):
#     """
#     判断一个短语是否包含地理位置信息。
#     :param phrase: 需要判断的短语
#     :return: 如果短语包含地理位置信息，则返回True，否则返回False
#     """
#     words = pseg.cut(phrase)
#     for word, flag in words:
#         if flag in ['ns', 'nt']:  # 'ns' 是地名的标记, 'nt' 是机构团体名的标记（有时也包含地名信息）
#             return True
#     return False

def detect_language(text):
    # 中文字符的Unicode范围
    chinese_re = re.compile(u'[\u4e00-\u9fff]+')
    # 英文字符的正则表达式
    english_re = re.compile(r'[A-Za-z]+')
    
    if chinese_re.search(text):
        return 'Chinese'
    elif english_re.search(text):
        return 'English'
    else:
        return 'Unknown'

def contains_geographic_location(text):
    """
    判断一个短语是否包含地理位置信息。
    :param phrase: 需要判断的短语
    :return: 如果短语包含地理位置信息，则返回True，否则返回False
    """
    doc = nlp(text)
    geographical_entities = [ent.text for ent in doc.ents if ent.label_ in ('GPE', 'LOC')]
    if (geographical_entities != []) & (detect_language(text)=='Chinese'):
        return True
    else:
        return False
    
# # # 名词
def is_complete_noun(phrase):
    words = list(pseg.cut(phrase))
    # 检查是否只有一个词且词性是名词
    if len(words) == 1 and words[0].flag.startswith('n'):
        return True
    return False

    
# # # main
for i in range(n):
    if (ooddata['cat'][i]=='无任何明确意图'):
        if (is_pure_number(ooddata['review'][i])) or (ooddata['review'][i]=='<number_pattern>'):
            ooddata['cat'][i]='纯数字'
        else :
            if is_composed_of_keywords(ooddata['review'][i], intonation_words):
                ooddata['cat'][i]='语气词'
            else:
                if contains_miswake_word(ooddata['review'][i], miswake_words):
                    ooddata['cat'][i]='误唤醒'
                else:
                    if contains_insult_word(ooddata['review'][i], insult_words):
                        ooddata['cat'][i]='骂傻'
                    else:
                        if contains_wake_word(ooddata['review'][i], wake_words):
                            ooddata['cat'][i]='疑似唤醒'
                        else:
                            if contains_exit_word(ooddata['review'][i],exit_words):
                                ooddata['cat'][i]='正常退出'
                            else:
                                if contains_science_word(ooddata['review'][i],science_words):
                                    ooddata['cat'][i]='科普'
                                else:
                                    if contains_geographic_location(ooddata['review'][i]):
                                        ooddata['cat'][i]='地理位置实体'
                                    else:
                                        if contains_EC_word(ooddata['review'][i],EC_words):
                                            ooddata['cat'][i]='EC'
                                            # print(ooddata['review'][i])
                                        else:
                                            if contains_phone_word(ooddata['review'][i],phone_words):
                                                ooddata['cat'][i]='电话'
                                            else:
                                                if contains_noentity_word(ooddata['review'][i],noentity_words):
                                                    ooddata['cat'][i]='无实体/截断/停顿'
                                                else:
                                                    if is_complete_noun(ooddata['review'][i]):
                                                        ooddata['cat'][i]='其他实体'

meaning=ooddata
# 输出为Excel文件
output_file = 'E://meaning_noentity.xlsx'
meaning.to_excel(output_file, index=False, engine='openpyxl')

print(f'Data saved to {output_file}')






