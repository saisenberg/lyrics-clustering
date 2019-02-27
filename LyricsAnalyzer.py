# Import dependencies
import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = None

import string

from collections import namedtuple
from helpers import getTopicWords
from gensim.models import Doc2Vec
from nltk.corpus import stopwords

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import contractions
contractions.contractions_dict["i'm"] = 'i am'
contractions.contractions_dict["i'd"] = 'i would'
contractions.contractions_dict["i'd've"] = 'i would have'
contractions.contractions_dict["i'll"] = 'i will'
contractions.contractions_dict["i'm"] = 'i am'
contractions.contractions_dict["i've"] = 'i have'


class LyricsAnalyzer():
    
    # Initialize object
    # min_ and max_df = affect TF-IDF
    # norm = affects normalization
    def __init__(self, lyrics_df, min_df=0.03, max_df=0.60, norm='l2'):
        self.lyrics_df = lyrics_df
        self.artists = list(sorted(lyrics_df.artist_name.unique()))
        
        self.min_df = min_df
        self.max_df = max_df
        self.norm = norm
        
        self.NMF_df = None
        self.LDA_df = None
        self.tagged_lyrics = None
        self.d2v = None
        self.tsne_df = None
        
        # Intialize with normalized lyrics_df
        self.tfidf_and_normalize()
        self.count_vectorize()
  
  
    
    ######### VECTORIZATION #########    
    
    # TFIDF and normalize all song lyrics (for NMF)
    def tfidf_and_normalize(self):
        df = self.lyrics_df
        
        # TFIDF
        tfidf_vect = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,1), min_df=self.min_df, max_df=self.max_df)
        self.tfidf_vectorizer = tfidf_vect
        freqs_df = tfidf_vect.fit_transform(df.cleaned_line)
        
        # Normalize
        normalized_df = normalize(freqs_df, norm=self.norm)
        self.normalized_df = normalized_df
    
    
    # CountVectorize all song lyrics (for LDA)
    def count_vectorize(self):
        df = self.lyrics_df
        count_vect = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1,1), min_df=self.min_df, max_df=self.max_df)
        self.count_vectorizer = count_vect
        freqs_df = count_vect.fit_transform(df.cleaned_line)
        self.freqs_df = freqs_df
    
    
    
    ######### TOPIC EXTRACTION #########
    
    # NMF on the TFIDF & normalized matrix
    # n_topics = number of topics to extract
    # n_words_per_topic = number of words per topic to return
    def get_nmf_topics(self, n_topics, n_words_per_topic=8):
        nmf = NMF(n_components=n_topics, init='nndsvd', random_state=1)
        nmf.fit(self.normalized_df)
        nmf_df = getTopicWords(vectorizer=self.tfidf_vectorizer, model=nmf, n_words_per_topic=n_words_per_topic, n_topics=None)
        self.NMF_df = nmf_df
        return(nmf_df)
    
    
    # LDA on the count-vectorized matrix
    # n_topics = number of topics to extract
    # n_words_per_topic = number of words per topic to return
    def get_lda_topics(self, n_topics, n_words_per_topic=8):
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=25, learning_method='online', learning_offset=50, random_state=1)
        lda.fit(self.freqs_df)
        lda_df = getTopicWords(vectorizer=self.count_vectorizer, model=lda, n_topics=n_topics, n_words_per_topic=n_words_per_topic)
        self.LDA_df = lda_df
        return(lda_df)
    
    
    
    ######### DOC2VEC #########
    
    # Tag all song lyrics
    # update_interval = frequency of progress messages
    def tag_lyrics(self, update_interval=100):
        tagged_lyrics = []
        namedLyrics = namedtuple('namedLyric', 'words tags')
        
        n_songs = len(self.lyrics_df.cleaned_line)
        for num, text in enumerate(self.lyrics_df.cleaned_line):
            if num % update_interval == 0:
                print(f'{num} / {n_songs} ({round(100*(num/n_songs), 2)}%)')
            text = text.split(' ')
            text = [word for word in text if word not in stopwords.words('english')]
            tags = [num]
            tagged_lyrics.append(namedLyrics(text, tags))
        
        self.tagged_lyrics = tagged_lyrics
    
    
    # Doc2Vec
    # vector_size, window, min_alpha, min_count, epochs = affect Doc2Vec
    # update_interval = frequency of progress messages (if necessary)
    def doc2vec(self, vector_size=300, window=5, min_alpha=0.025, min_count=15, epochs=25, update_interval=100):
        if self.tagged_lyrics==None:
            self.tag_lyrics(update_interval)
        
        d2v = Doc2Vec(self.tagged_lyrics, vector_size=vector_size, window=window, min_alpha=min_alpha, min_count=min_count, epochs=epochs, dm=1)
        self.d2v = d2v
    
    
    # TSNE from Doc2Vec results
    # return_df = whether to return the t-SNE dataframe (will be saved to self regardless)
    # perplexity, n_iter = affect t-SNE
    def tsne_from_d2v(self, return_df=True, perplexity=40, n_iter=400):
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=1)
        
        if self.d2v == None:
            print('ERROR - please run doc2vec(...) first!')
        df_d2v = pd.DataFrame(self.d2v.docvecs.vectors_docs)
        
        tsne_df = pd.DataFrame(tsne.fit_transform(df_d2v))
        tsne_df = tsne_df.rename(columns={0:'TSNE1', 1:'TSNE2'})
        tsne_df['song_name'] = self.lyrics_df.song_name
        tsne_df['artist_name'] = self.lyrics_df.artist_name
        tsne_df['genre'] = self.lyrics_df.genre
        
        self.tsne_df = tsne_df
        if return_df:
            return(tsne_df)
    
    
    # Save TSNE dataframe to .csv
    # folder_path = path to folder in which to save t-SNE dataframe
    def tsne_to_csv(self, folder_path):
        if self.d2v == None:
            print('ERROR - please run tsne_from_d2v(...) first!')
        
        new_file_name = '_x_'.join(self.artists)
        new_file_name = new_file_name.replace(' ', '')
        new_file_name = ''.join([value for value in new_file_name if value not in string.punctuation.replace('_', '')])
        
        self.tsne_df.to_csv(f'{folder_path}\\{new_file_name}.csv', index=False)
        print('TSNE dataframe saved to CSV!')
    