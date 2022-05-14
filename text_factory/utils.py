from collections import Counter
from typing import List, Tuple, Union

import networkx as nx
from networkx import (
    spring_layout,
    spectral_layout,
    circular_layout,
    random_layout,
    spectral_layout,
    spiral_layout,
    shell_layout,
    planar_layout,
    bipartite_layout,
    kamada_kawai_layout,
)
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


def tf_idf(corpus: List[str]):
    tfidfv = TfidfVectorizer().fit(corpus)
    array = tfidfv.transform(corpus).toarray()
    index = tfidfv.vocabulary_

    return array, index


def create_wordcloud(frequency: dict):
    wc = WordCloud(
        background_color="white",
        width=1000,
        height=1000,
        max_words=100,
        max_font_size=300,
    )

    wc.generate_from_frequencies(frequency)

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(wc.generate_from_frequencies(frequency))
    plt.savefig("word_cloud")


def run_network(
    frequency: Union[Counter, dict],
    layout: Union[
        spring_layout,
        spectral_layout,
        circular_layout,
        random_layout,
        spectral_layout,
        spiral_layout,
        shell_layout,
        planar_layout,
        bipartite_layout,
        kamada_kawai_layout,
    ] = spring_layout,
    figsize: Tuple[int, int] = (80, 80),
):
    G = nx.Graph()

    edge_list = []
    keywords = list(frequency)
    num_keyword = len(keywords)
    for i in range(num_keyword - 1):
        for j in range(i + 1, num_keyword):
            edge_list += [tuple(sorted([keywords[i], keywords[j]]))]
    edges = list(Counter(edge_list).items())

    G = nx.Graph((x, y, {'weight': v}) for (x, y), v in edges)

    nx.Graph()
    plt.figure(figsize=figsize)
    pos = layout(G)
    nx.draw_networkx_nodes(
        G, pos, node_shape="o", node_color='#BB78FF', node_size=3000
    )
    nx.draw_networkx_edges(G, pos, style='solid', width=5, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=45)
    plt.savefig("networkx_graph")
    plt.show()
