import jieba
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from article_loader import ArticleLoader

article_loader =  ArticleLoader('chinese_zh-TW_corpora.yaml')
articles = article_loader.load()

articles = [re.sub('[^\u4E00-\u9FA5A-Za-z0-9]', ' ', article)
    for article in articles]

jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('dict.txt')

def tokenizer(sentence, flit=[' '], **kwargs):
    return [token for token in jieba.cut(sentence, **kwargs)
        if token not in flit]

def similarity(articles, **kwargs):
    tfid_vectorizer = TfidfVectorizer(tokenizer=tokenizer, **kwargs)
    matrix = tfid_vectorizer.fit_transform(articles)
    return cosine_similarity(matrix)[0, 1]


print(f'1-gram TF-IDF cosine similarity: {similarity(articles)}')
print(f'2-gram + 3-gram TF-IDF cosine similarity: \
    {similarity(articles, ngram_range=(2, 3))}')