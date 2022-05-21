import string
from collections import Counter
from typing import List, Union

import nltk
import pandas as pd
from konlpy.tag import Komoran
from nltk.tokenize import TweetTokenizer

nltk.download('punkt')


class BasePreprocessor:
    def get_count(text: str, k: int = 100):
        count = Counter(text)
        return count.most_common(k)


class KoreanPreprocessor(BasePreprocessor):
    def __init__(self):
        self.komoran = Komoran()

    def extract_keyword(self, text: str):
        return set(
            [nouns for nouns in self.komoran.nouns(text) if len(nouns) > 1]
        )


class EnglishPreprocessor(BasePreprocessor):
    """
    Note that you must download relevant dataset before using classes from nltk.
    """

    def __init__(self):
        pass

    def tokenize(
        self, tokenizer, text_series: pd.Series, pos_list: List[str] = ["NN"]
    ) -> pd.Series:
        def _tokenize(tweet: str):
            try:
                tokens = tokenizer.tokenize(tweet)
            except AttributeError:
                tokens = tokenizer(tweet)

            tags = nltk.pos_tag(tokens)

            filtered_tokens = []
            for word, pos in tags:
                token = word.lower()
                if pos not in pos_list:
                    continue

                if token in string.punctuation:
                    continue
                elif len(token) <= 3:
                    continue
                elif token.startswith(('http', 'www')):
                    continue
                else:
                    filtered_tokens.append(token)

            return " ".join(filtered_tokens)

        return text_series.apply(_tokenize)

    def tweet_tokenize(
        self, text_series: pd.Series, pos_list: List[str] = ["NN"]
    ) -> pd.Series:
        tokenizer = TweetTokenizer(
            preserve_case=True, reduce_len=False, strip_handles=False
        )

        return self.tokenize(tokenizer, text_series, pos_list)

    def lemmatize(
        self, lemmatizer, token_series: Union[List[str], pd.Series]
    ) -> pd.Series:
        # TODO: support List[str]
        def _lemmatize(tokens: Union[List[str], pd.Series]):
            return " ".join(map(lemmatizer.lemmatize, tokens))

        return token_series.apply(lambda x: x.split()).apply(_lemmatize)

    def multiword_tokenize(
        self,
        text_series: pd.Series,
        multiwords: List[List[str]],
    ) -> pd.Series:
        mwe_tokenizer = nltk.tokenize.MWETokenizer(multiwords)
        return text_series.apply(
            lambda text: mwe_tokenizer.tokenize(text.split())
        )
