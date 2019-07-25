import pandas as pd
import re
import jieba
import word2vec


df = pd.read_csv('PTT_Gossiping_20161105_20161112_post.csv')

content = ' '.join(df['POST_CONTENT'])
content = re.sub('[^\u4E00-\u9FA5]', ' ', content)

jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('custom_dict')
content = ' '.join(jieba.cut(content))

with open('ptt-post-content', 'w') as fd:
    fd.write(content)

word2vec.word2vec('ptt-post-content', 'ptt-post-content.bin', size=100,
                   verbose=True)

