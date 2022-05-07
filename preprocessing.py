import string
from collections import Counter

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

    def extract_keyword(text: str):
        return set(
            [nouns for nouns in self.komoran.nouns(text) if len(nouns) > 1]
        )


class EnglishPreprocessor(BasePreprocessor):
    def __init__(self):
        self.tweet_tokenizer = TweetTokenizer(
            preserve_case=True, reduce_len=False, strip_handles=False
        )

    def tweet_tokenize(self, tweet_series: pd.Series):
        def tokenize_tweet(tweet: str):
            tokens = self.tweet_tokenizer.tokenize(tweet)
            filtered_tokens = []
            for token in tokens:
                if token in string.punctuation:
                    continue
                elif len(token) <= 3:
                    continue
                elif token.startswith(('http', 'www')):
                    continue
                else:
                    filtered_tokens.append(token)

            return " ".join(filtered_tokens)

        return tweet_series.apply(tokenize_tweet)
