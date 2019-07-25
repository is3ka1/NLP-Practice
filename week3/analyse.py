import word2vec

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

from sklearn.decomposition import PCA


model = word2vec.load('ptt-post-content.bin')

pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(model.vectors)

target_words = '媒體', '網軍', '黨', '台灣'

indices_and_distances_by_words = [model.cosine(word, n=8) 
                                  for word in target_words]

data_to_draw = [(model.vocab[indices], vectors_pca[indices])
                for indices, _ in indices_and_distances_by_words]

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['figure.autolayout'] = True

fig = plt.figure(figsize=(9, 6))
axes = fig.add_subplot()

for i, data in enumerate(data_to_draw):
    for word, vectors in zip(*data):
        axes.text(*vectors, word,ha='center', va='center', color=f'C{i}')

XYs = np.stack([vectors for _, vectors in data_to_draw]).reshape(-1, 2)

axes.axis([XYs[:, 0].min() - 0.1, XYs[:, 0].max() + 0.1,
           XYs[:, 1].min() - 0.1, XYs[:, 1].max() + 0.1])

patches = [mpatches.Patch(color=f'C{i}', label=f'{word}') 
           for i, word in enumerate(target_words)]

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., handles=patches)
plt.savefig('target_words_top8.png')


for i, indices_and_distances in enumerate(indices_and_distances_by_words):
    print(target_words[i])
    print(model.generate_response(*indices_and_distances))