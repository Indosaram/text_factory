import pyLDAvis.gensim_models
from gensim import corpora, models


class TopicModeling:
    def __init__(self, tokenized_text: str):
        self.tokenized_text = tokenized_text
        self.dictionary = corpora.Dictionary(self.tokenized_text)
        self.corpus = [
            self.dictionary.doc2bow(text) for text in self.tokenized_text
        ]

    def lda(self, num_topics: int = 20):
        return models.ldamodel.LdaModel(
            self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            passes=15,
        )

    def get_topics(
        self, ldamodel: models.ldamodel.LdaModel, num_words: int = 4
    ):
        topics = ldamodel.print_topics(num_words=num_words)
        return topics

    def topic_visualize(self, ldamodel: models.ldamodel.LdaModel):
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim_models.prepare(
            ldamodel, self.corpus, self.dictionary
        )
        pyLDAvis.display(vis)
