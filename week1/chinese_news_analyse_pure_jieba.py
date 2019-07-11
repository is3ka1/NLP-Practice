import jieba
import jieba.analyse

from math import sqrt

from article_loader import ArticleLoader

article_loader =  ArticleLoader('chinese_zh-TW_corpora.yaml')
articles = article_loader.load()

jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('dict.txt')

term_with_tfidfs = [jieba.analyse.tfidf(article, topK=False, withWeight=True)
    for article in articles]

term_2_tfidfs = [dict(twt) for twt in term_with_tfidfs]

features = list(
    set.union(*[set(list(zip(*twt))[0]) for twt in term_with_tfidfs]))

matrix = [[term_2_tfidf.get(term, 0.) for term in features]
    for term_2_tfidf in term_2_tfidfs]

def distance(vector):
    return sqrt(sum(c ** 2 for c in vector))

def dot(vectors):
    return sum(a * b for a, b in zip(*vectors))

def cosine_similarity(vectors):
    return dot(vectors) / (distance(vectors[0]) * distance(vectors[1]))

print(f'1-gram TF-IDF cosine similarity: {cosine_similarity(matrix)}')

from wordcloud_process import wordcloud_img

for i, t2t in enumerate(term_2_tfidfs):
    wordcloud_img(t2t, f'image/Chinese_News_pure_jieba_{i}.png')