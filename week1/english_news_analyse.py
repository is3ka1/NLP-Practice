from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud_process import wordcloud_img

from article_loader import ArticleLoader

article_loader =  ArticleLoader('english_corpora.yaml')
articles = article_loader.load()

def feature_and_matrix(articles, **kwargs):
    tfid_vectorizer = TfidfVectorizer(**kwargs)
    matrix = tfid_vectorizer.fit_transform(articles)
    return tfid_vectorizer.get_feature_names(), matrix.A

def produce_wordcloud(features, vector, file_name):
    wordcloud_img(dict((k, v) for k, v in zip(features, vector) if v != 0.),
                  file_name)


features, matrix = feature_and_matrix(articles)
for i, vector in enumerate(matrix):
    produce_wordcloud(features, vector, f'image/English_News_{i}.png')
print(f'1-gram TF-IDF cosine similarity: {cosine_similarity(matrix)[0, 1]}')


features, matrix = feature_and_matrix(articles, ngram_range=(2, 3))
for i, vector in enumerate(matrix):
    produce_wordcloud(features, vector, f'image/English_News_{i}_2+3-gram.png')
print(f'2-gram + 3-gram TF-IDF cosine similarity: \
    {cosine_similarity(matrix)[0, 1]}')