import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt



def plot_barplot(df, n):
    '''Returns None, plots barplot with n most frequent words.'''
    plt.figure(figsize=(12,8))
    plt.title("Frequencies of the %s most frequent words after TF-IDF" %n)
    sns.barplot(data=df[:n].T).set_xticklabels(labels=df[:n].index,rotation=60)


def plot_wordcloud_tf_idf(df, rating):
    '''Returns None, plots wordcloud from df.'''
    # transforms data frame to dictionary
    dict_words_tf_idf = df[df['tf_idf_mean'] != 0].to_dict()['tf_idf_mean']
    print("test")
    # create wordcloud
    if rating == "Good":
        mask = np.array(Image.open("../images/mask2.PNG"))
        cmap = "viridis"
        print("good")
    elif rating == "Bad":
        mask = np.array(Image.open("../images/mask3.PNG"))
        cmap = "inferno"
        print("bad")
    wordcloud = WordCloud(height=600, width=800, background_color="White",
                          max_words=200, mask=mask, colormap=cmap)
    wordcloud.generate_from_frequencies(frequencies=dict_words_tf_idf)
    # plot wordcloud
    plt.figure(figsize=(12,16))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Wordcloud after TF-IDF")
    plt.axis("off")
    plt.show()
    

def tf_idf(corpus, n=15, barplot=True, wordcloud=False, rating="Bad"):
    '''
    Returns TF-IDF matrix based on clean corpus.
    
    Parameters
    ----------
    corpus: list
        list of sentences to be processsed
    n: int
        number of words to plot in barplot
    barplot: Bool
        if True, plots a barplot with most frequent words
    wordcloud: Bool
        if True, plots a wordcloud with most frequent words
    
    Returns
    -------
    df_tfidf: pandas.DataFrame
        matrix of relative frequency of words in corpus
    
    '''
    # apply TF-IDF to corpus
    untokenized_corpus = [" ".join(words) for words in corpus]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    vect_corpus = vectorizer.fit_transform(untokenized_corpus)
    # retrieve words
    feature_names = np.array(vectorizer.get_feature_names())
    # represent data as a data frame
    df_tfidf = pd.DataFrame(vect_corpus.todense(), columns = feature_names)
    
    # calculate corpus relative frequence of words for representation 
    df_tfidf_mean = df_tfidf.mean().sort_values(ascending=False).to_frame(
        name='tf_idf_mean'
    ) 
    # plot 
    if barplot == True:
        plot_barplot(df_tfidf_mean, n)
    if wordcloud == True:
        plot_wordcloud_tf_idf(df_tfidf_mean, rating)

    return df_tfidf