# import libraries
import pandas as pd
import numpy as np
import sys


# nltk functions
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download("punkt")
nltk.download("stopwords")

# other textpreprocessing functions
import os
import json
import re
import string
from src.utils.processing_utils import get_char_sentence, get_char_corpus


class TextPreprocessor:
    """
    Class to preprocess the text of the reviews obtained from Trip Advisor
    Parameters
    ----------
    df_to_clean: pd.DataFrame
        df containg reviews
    column_to_clean: str
        name of the column of the df containing reviews
    chars: str
        list of characters that are acceptable. All other characters will
        not be kept.
    Attributes
    ----------
    df_to_clean: pd.DataFrame
        df containg reviews
    column_to_clean: str
        name of the column of the df containing reviews
    chars: str
        list of characters that are acceptable
    corpus: list
        list containing all reviews

    """

    def __init__(
        self,
        df_to_clean,
        column_to_clean="review_content",
        chars=string.ascii_lowercase + string.digits + " ",
    ):
        self.df_to_clean = df_to_clean
        self.column_to_clean = column_to_clean
        self.corpus = self._corpus_creator()
        self.chars = chars

    def _corpus_creator(self):
        """Returns a list with all reviews in the column of df_to_clean."""
        corpus = self.df_to_clean[self.column_to_clean].tolist()

        return corpus

    def _lowercase_transformer(self):
        """Returns corpus with only lower-case characters."""
        corpus = [review.lower() for review in self.corpus]

        return corpus

    def _digit_transformer(self):
        """Returns corpus with written digits."""
        transform_dict = {
            "1": "one ",
            "2": "two ",
            "3": "three ",
            "4": "four ",
            "5": "five ",
            "6": "six ",
            "7": "seven ",
            "8": "eight ",
            "9": "nine ",
            "0": "zero ",
        }

        transformer = str.maketrans(transform_dict)

        return [review.translate(transformer) for review in self.corpus]

    def decontractor(self, sentence):
        """Returns sentence without contractions"""

        # punctuation mistake
        sentence = re.sub(r"’", "'", sentence)

        # specific
        sentence = re.sub(r"won\'t", "will not", sentence)
        sentence = re.sub(r"can\'t", "can not", sentence)

        # general
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)

        return sentence

    def _accent_transformer(self):
        """Returns corpus without accents."""
        transform_dict = {
            "ú": "u",
            "î": "i",
            "í": "i",
            "è": "e",
            "ö": "o",
            "ı": "i",
            "é": "e",
            "ï": "i",
            "ê": "e",
            "ť": "t",
            "ü": "u",
            "ó": "o",
            "ñ": "n",
            "ć": "c",
            "ù": "u",
            "ț": "t",
            "û": "u",
            "â": "a",
            "ô": "o",
            "à": "a",
            "á": "a",
            "ĺ": "l",
            "ç": "c",
            "ď": "d",
            "е": "e",
        }
        transformer = str.maketrans(transform_dict)

        return [review.translate(transformer) for review in self.corpus]

    def _char_filter(self):
        """Returns corpus with only the allowed characters"""
        corpus_chars = get_char_corpus(self.corpus)
        set_chars = get_char_sentence(self.chars)

        unwanted = corpus_chars - set_chars  # set difference

        transform_dict = {char: " " for char in unwanted}
        transformer = str.maketrans(transform_dict)

        return [review.translate(transformer) for review in self.corpus]

    def _ngrams(self, review, n):
        """Returns n-grams tokenized corpus."""
        n_grams = ngrams(nltk.tokenize.word_tokenize(review), n)

        return [" ".join(grams) for grams in n_grams]

    def _tokenizer(self, n_grams):
        """Returns tokenized corpus."""
        if n_grams:
            corpus = [
                word_tokenize(review)
                + self._ngrams(review, 2)
                + self._ngrams(review, 3)
                for review in self.corpus
            ]
        else:
            corpus = [word_tokenize(review) for review in self.corpus]

        return corpus

    def _stopword_remover(self):
        """Returns corpus without stopwords."""
        # create list of stopwords to remove
        stopword_list = stopwords.words("english")

        with open(
            os.path.join("..", "data", "resources", "cities_and_countries.json"), "rb"
        ) as f:
            cities_in_reviews = json.load(f)
        cities = list(cities_in_reviews.keys())
        countries = list(set(cities_in_reviews.values()))
        print(f"cities: {cities}")
        print(f"countries: {countries}")
        stopword_list += cities
        stopword_list += countries
        stopword_list += ["verified", "zero", "flight", "flights", "trip", "good", "seven"]

        # remove stopwords
        corpus = [
            [token for token in tokenized_review if token not in stopword_list]
            for tokenized_review in self.corpus
        ]

        return corpus

    def transform(self, tokenize=True, remove_stopwords=False, n_grams=False):
        # call lowercase
        self.corpus = self._lowercase_transformer()
        # call digit_tranformer
        self.corpus = self._digit_transformer()
        # call decontractor
        self.corpus = [self.decontractor(review) for review in self.corpus]
        # call accent_tranformer
        self.corpus = self._accent_transformer()
        # call character filter
        self.corpus = self._char_filter()

        if tokenize:
            # call tokenizer
            self.corpus = self._tokenizer(n_grams)

        if remove_stopwords:
            if tokenize:
                # call stopword remover
                self.corpus = self._stopword_remover()
            else:
                raise ValueError("To remove stopwords, tokenize must be True")


# function to split reviews by sentence
def split_reviews_per_sentence(reviews):
    reviews["review_sentences"] = reviews["review_content"].progress_apply(
        lambda rvw: nltk.sent_tokenize(rvw)
    )
    return reviews
