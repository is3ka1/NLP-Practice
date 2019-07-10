from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from article_loader import ArticleLoader

article_loader =  ArticleLoader('english_corpora.yaml')
articles = article_loader.load()

tfid_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
matrix = tfid_vectorizer.fit_transform(articles)

print(f'Similarity: {cosine_similarity(matrix)[0, 1]}')
