from wordcloud import WordCloud

wc = WordCloud(font_path='NotoSansCJKtc-Regular.otf', width=1024, height=720, background_color='white')

def wordcloud_img(freqs: dict, path):
    wc.generate_from_frequencies(freqs)
    wc.to_file(path)