import jieba
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from article_loader import ArticleLoader

article_loader =  ArticleLoader('chinese_zh-TW_corpora.yaml')
articles = article_loader.load()

articles = [re.sub('[^\u4E00-\u9FA5A-Za-z0-9]', ' ', article) for article in articles]

jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('dict.txt')

def tokenizer(sentence, flit=' ', **kwargs):
    for token in jieba.cut(sentence, **kwargs):
        if not token == flit:
            yield token

tfid_vectorizer = TfidfVectorizer(tokenizer=tokenizer)
matrix = tfid_vectorizer.fit_transform(articles)

print(f'Similarity: {cosine_similarity(matrix)[0, 1]}')


# tokens_space = set.union(*[set(tokenizer(article)) for article in articles])
# print(matrix.A.shape[1] == len(tokens_space))                     # True
# print(set(tfid_vectorizer.get_feature_names()) == tokens_space)   # True
