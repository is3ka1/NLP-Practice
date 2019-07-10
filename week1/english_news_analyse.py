from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from article_loader import ArticleLoader

article_loader =  ArticleLoader('english_corpora.yaml')
articles = article_loader.load()

def similarity(articles, **kwargs):
    tfid_vectorizer = TfidfVectorizer(**kwargs)
    matrix = tfid_vectorizer.fit_transform(articles)
    return cosine_similarity(matrix)[0, 1]


print(f'1-gram TF-IDF cosine similarity: {similarity(articles)}')

print(f'2-gram + 3-gram TF-IDF cosine similarity: \
    {similarity(articles, ngram_range=(2, 3))}')
