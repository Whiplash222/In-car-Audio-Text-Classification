# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:41:43 2024

@author: YUE
"""

# # rule

import rulecheck

rulecheck.main()

# # cnn

import cnn_update

cnn_update.main()

# # mix
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
# Load dataset
df_music = pd.read_excel('./predictions_music.xlsx')
df_vehicle = pd.read_excel('./predictions_vehicle.xlsx')
df_rulecheck = pd.read_excel('./meaning_noentity.xlsx')
df_final=df_rulecheck

df_music['maxprob']

row_indices_music = df_music.index[(df_music['maxprob'] > 0.5) & (df_music['y_pred'] == 0)].tolist()
row_indices_vehicle = df_vehicle.index[(df_vehicle['maxprob'] > 0.99989) & (df_vehicle['y_pred'] == 1)].tolist()

df_final.loc[row_indices_vehicle, 'cat'] = df_final.loc[row_indices_vehicle, 'cat'].apply(lambda x: '控单实体/完整句' if ((x == '无任何明确意图') or (x == '其他实体')) else x)
df_final.loc[row_indices_music, 'cat'] = df_final.loc[row_indices_music, 'cat'].apply(lambda x: '媒体单实体/完整句' if ((x == '无任何明确意图') or (x == '其他实体')) else x)

# 输出为Excel文件
output_file = './classification_results.xlsx'
df_final.to_excel(output_file, index=False, engine='openpyxl')

print(f'Data saved to {output_file}')





