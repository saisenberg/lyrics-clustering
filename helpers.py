import pandas as pd
import re
import string

from nltk.corpus import stopwords

import contractions
contractions.contractions_dict["i'm"] = 'i am'
contractions.contractions_dict["i'd"] = 'i would'
contractions.contractions_dict["i'd've"] = 'i would have'
contractions.contractions_dict["i'll"] = 'i will'
contractions.contractions_dict["i'm"] = 'i am'
contractions.contractions_dict["i've"] = 'i have'

# Collect TOPIC WORDS of each NMF component
def getTopicWords(vectorizer, model, n_words_per_topic, n_topics=None):
    if not n_topics:
        n_topics = model.components_.shape[0]
    
    feature_names = vectorizer.get_feature_names()
    word_dict = {}
    
    for i in range(n_topics):
        col_num = i+1
        top_words_indices = model.components_[i].argsort()[-n_words_per_topic:][::-1]
        top_words = [feature_names[index] for index in top_words_indices]
        word_dict[f'Topic {col_num}'] = top_words
    
    words_df = pd.DataFrame(word_dict).transpose()
    col_names = []
    for col in words_df.columns:
        col_names.append(f'word_{col+1}')
    words_df.columns = col_names
    words_df = words_df.transpose()
    
    return(words_df)




    
# Preprocess lyrics
def preprocessText(text, remove_stops=False):
    
    # Remove everything between hard brackets
    text = re.sub(pattern="\[.+?\]( )?", repl='', string=text)

    # Change "walkin'" to "walking", for example
    text = re.sub(pattern="n\\\' ", repl='ng ', string=text)

    # Remove x4 and (x4), for example
    text = re.sub(pattern="(\()?x\d+(\))?", repl=' ', string=text)

    # Fix apostrophe issues
    text = re.sub(pattern="\\x91", repl="'", string=text)
    text = re.sub(pattern="\\x92", repl="'", string=text)
    text = re.sub(pattern="<u\+0092>", repl="'", string=text)
    
    # Make lowercase
    text = text.lower()
    
    # Special cases/words
    text = re.sub(pattern="'til", repl="til", string=text)
    text = re.sub(pattern="'til", repl="til", string=text)
    text = re.sub(pattern="gon'", repl="gon", string=text)

    # Remove \n from beginning
    text = re.sub(pattern='^\n', repl='', string=text)

    # Strip , ! ?, : and remaining \n and \r from lyrics
    text = ''.join([char.strip(",!?:") for char in text])
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\r', ' ')

    # Remove contractions
    text = contractions.fix(text)

    # Replace hyphens with spaces
    text = re.sub(pattern="-", repl=" ", string=text)
    
    # Remove remaining punctuation
    punc = string.punctuation
    text = ''.join([char for char in text if char not in punc])

    # Remove stopwords
    if remove_stops:
        stops = stopwords.words('english')
        text = ' '.join([word for word in text.split(' ') if word not in stops])
    
    # Remove double spaces and beginning/trailing whitespace
    text = re.sub(pattern='( ){2,}', repl=' ', string=text)
    text = text.strip()
    
    return(text)





# Collect lyrics of specified artists
def prepareLyricsFrame(artist_dict, path):
    
    # Initialize dataframe
    lyrics_df = pd.DataFrame()
    
    # For every genre
    for genre in artist_dict.keys():
        genre_path = f'{path}\\{genre}\\'
        
        # For every artist
        if len(artist_dict[genre]) > 0:
            for artist in artist_dict[genre]:
                artist_df = pd.read_csv(genre_path + '\\corpus_' + artist + '.csv')
                artist_df['genre'] = genre
                lyrics_df = pd.concat([lyrics_df, artist_df])
    
    # Remove unneeded songs
    lyrics_df = lyrics_df[~lyrics_df.line.isnull()]   # NaNs
    lyrics_df['song_name'] = lyrics_df.song_name.apply(lambda x: x.lower())
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('remix')].reset_index(drop=True)     # no remixes
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('live in')].reset_index(drop=True)     # no live performances (1)
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('live on')].reset_index(drop=True)     # no live performances (2)
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('live from')].reset_index(drop=True)     # no live performances (3)
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('version')].reset_index(drop=True)     # no 'version's
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('freestyle')].reset_index(drop=True)     # no freestyles
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('translation')].reset_index(drop=True)     # no translations
    lyrics_df = lyrics_df[~lyrics_df.song_name.str.contains('statement')].reset_index(drop=True)     # no 'statements'
    lyrics_df = lyrics_df[~lyrics_df.line.str.contains('snippet')].reset_index(drop=True)     # no lyric snippets
    
    # Preprocess lyrics
    lyrics_df['song_name'] = lyrics_df.song_name.apply(lambda x: re.sub(string=x, pattern='\\*', repl=''))   # Remove asterisks from song name
    lyrics_df['cleaned_line'] = lyrics_df.line.apply(lambda x: '. '.join(x.split('\\n')))    # Split and rejoin lyrics
    lyrics_df['cleaned_line'] = lyrics_df.cleaned_line.apply(lambda x: preprocessText(x))    # Preprocess lyrics
    
    return(lyrics_df)
