import os

# Set your working directory
os.chdir(r'[SET WORKING DIRECTORY]')

from helpers import prepareLyricsFrame
from LyricsAnalyzer import LyricsAnalyzer
from params import artist_dict, corpus_path, tsne_path


# Prepare a dataframe with lyrics from all artist_dict artists
lyrics_df = prepareLyricsFrame(artist_dict = artist_dict, path = corpus_path)

# Create LyricsAnalyzer object
la = LyricsAnalyzer(lyrics_df)

# Collect topics (LDA)
lda_topics = la.get_lda_topics(n_topics=8)

# Collect topics (NMF)
nmf_topics = la.get_nmf_topics(n_topics=8)

# Create Doc2Vec vectors of every song - may take a few minutes
la.doc2vec()

# Convert document vectors to two dimensions
la.tsne_from_d2v(return_df=False)

# Write to csv
la.tsne_to_csv(tsne_path)