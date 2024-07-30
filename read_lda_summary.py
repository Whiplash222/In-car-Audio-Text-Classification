# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:23:16 2024

@author: pengyue
"""

import pandas as pd
import re
import jieba
import nltk 
from nltk.corpus import stopwords
from gensim import corpora
import numpy as np
from gensim.models import LdaModel

stoplist = []
for line in open('E://AISpeech/stop_w.txt', 'r',encoding='utf-8').readlines():
    stoplist.append(line.strip())

# 去噪函数
def remove_noise(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 只保留中文字符
    return text

# 分词和去除停用词函数
def preprocess_text(text):
    text = remove_noise(text)
    words = jieba.cut(text)
    words = [word for word in words if word not in stoplist] # 去掉停止词
    return words

# 读取Excel文件
np.random.seed(22)
df = pd.read_excel('E://AISpeech/OOD data.xlsx')
random_rows = df.sample(n=100)

# 初始化存储内容和标签的列表
content_all = []
label_all = []

# print(df)
# print(df.iterrows())

# 遍历数据并进行预处理
for index, row in df.iterrows():
    content = preprocess_text(str(row[2]))
    # print(content)
    label = row[3]
    # print(label)
    content_all.append(content)
    label_all.append(label)
    
toremove=[]
content_nan = [x for x in content_all if x != toremove]
# 构建词典
dictionary = corpora.Dictionary(content_nan)
# 移除出现频率极低或极高的词
dictionary.filter_extremes(no_below=1, no_above=0.5)
dictionary.filter_tokens(bad_ids=[dictionary.token2id.get('')])
dictionary.compactify()

# 构建语料库
corpus = [dictionary.doc2bow(text) for text in content_nan]

# 打印示例
print(content_nan[:5])
print(dictionary.token2id)
print(corpus[:5])

# Set training parameters.
num_topics = 6
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

topic_list=model.print_topics()
# g = open('C://Users/pengyue/text_aispeech/Lda_new_stopw12.csv', 'w',encoding='utf-8')
# g.write(str(topic_list))

for topic in topic_list:
    print(topic)
# 打印LDA模型结果
for idx, topic in model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# for i, doc_bow in enumerate(corpus):
#     topics = model.get_document_topics(doc_bow)
#     print(f"Document {i+1} topic distribution:")
#     for topic_num, prob in topics:
#         print(f"  Topic {topic_num}: {prob:.4f}")


# random_rows = df.sample(n=100)
print('select samples!!!')
print(random_rows)

# 初始化存储内容和标签的列表
content_select = []
label_select = []

# print(df)
# print(df.iterrows())

# 遍历数据并进行预处理
for index, row in random_rows.iterrows():
    content = preprocess_text(str(row[2]))
    # print(content)
    # print(label)
    content_select.append(content)
print(content_select)

results = []
for new_doc in content_select:
    new_doc_bow = dictionary.doc2bow(new_doc)
    topics = model.get_document_topics(new_doc_bow)
    # 将结果添加到列表中
    results.append([new_doc] + [prob for _, prob in sorted(topics)])

# 将结果转换为DataFrame
df = pd.DataFrame(results, columns=['Document'] + [f'Topic_{i}' for i in range(num_topics)])

# 保存结果为CSV文件
df.to_csv('E://lda_results.csv', index=False)

print("Results saved to lda_results.csv")
# import pyLDAvis.gensim_models as gensimvis
# import pyLDAvis
# Visualize the topics
# vis_data = gensimvis.prepare(model, corpus, dictionary)
# pyLDAvis.display(vis_data)

# import pyLDAvis.gensim
# import pyLDAvis

# data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
# pyLDAvis.save_html(data,'C://Users/pengyue/text_aispeech/Lda_visual results_new_stopw12.html')