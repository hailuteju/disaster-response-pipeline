import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import (BaseEstimator, TransformerMixin)

nltk.download(['punkt', 'wordnet', 'words', 'stopwords',
               'averaged_perceptron_tagger'])

stop_words = stopwords.words("english")


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A custom estimator class that inherits from the `BaseEstimator` and
    `TransformerMixin` super classes. This custom estimator has three methods:
    strting_verb, fit, and transform methods. The starting_verb method takes a text as
    an input and returns a Boolean value depending on whether or not the text starts
    with a verb or not.
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0: continue
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    """
    This function takes a text as input, tokenizes the text, lemmatizes the
    resulting tokens, and returns a list of clean tokens after also removing
    stop words.

    Args:
      text (str): text to be tokenized

    Returns:
      clean_tokens: list of cleaned tokens

    """
    regex_url = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    text = regex_url.sub("urlplaceholder", text)

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip()
                    for tok in tokens if tok not in stop_words]

    return clean_tokens