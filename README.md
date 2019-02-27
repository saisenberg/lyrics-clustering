## Lyrics, Pt. 3: Rap Song Clustering with Doc2Vec

This project creates embeddings of song lyrics with Doc2Vec, reduces the resulting dimensionality with t-SNE, and compares artist-by-artist song clusters.

A full description of the project can be found at [**saisenberg.com**](https://saisenberg.com/projects/lyrics-clustering.html).

### Getting started

#### Prerequisite software

* Python

#### Prerequisite libraries

* Python:

```
contractions, collections, gensim, nltk, pandas, re, sklearn, string (```install any missing libraries with !pip install [library name]```)
```

### Instructions for use

- Change paths in *params.py* as appropriate.

- Update artist dictionary in *params.py* as appropriate.

- Change working directory in *run.py* as appropriate, and run entire file. Note that additional parameters are available for many *LyricsAnalyzer* methods; see *LyricsAnalyzer.py* for further details on available options.


### Author

* **Sam Isenberg** - [saisenberg.com](https://saisenberg.com) | [github.com/saisenberg](https://github.com/saisenberg)


### License

This project is licensed under the MIT License - see the *LICENSE.md* file for details.

### Acknowledgements

* [Genius](http://genius.com)