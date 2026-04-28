import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def matrix_to_digraph(matrix, weight_col="weight", min_weight=0):
    matrix = matrix.copy()
    matrix.index.name = "source_sector"
    matrix.columns.name = "target_sector"

    edges = matrix.stack().reset_index(name=weight_col)
    edges = edges[edges[weight_col] > min_weight]

    return nx.from_pandas_edgelist(
        edges,
        source="source_sector",
        target="target_sector",
        edge_attr=weight_col,
        create_using=nx.DiGraph(),
    )


def compute_in_strength(G, weight="weight"):
    return pd.Series(dict(G.in_degree(weight=weight)), name="in_strength")


def compute_out_strength(G, weight="weight"):
    return pd.Series(dict(G.out_degree(weight=weight)), name="out_strength")


def compute_eigenvector_centrality(G, weight="weight"):
    return pd.Series(
        nx.eigenvector_centrality(G, weight=weight, max_iter=1000),
        name="eigenvector_centrality",
    )


def compute_reverse_eigenvector_centrality(G, weight="weight"):
    return pd.Series(
        nx.eigenvector_centrality(G.reverse(), weight=weight, max_iter=1000),
        name="reverse_eigenvector_centrality",
    )


def compute_pagerank(G, weight="weight", alpha=0.85, max_iter=1000):
    try:
        pagerank_values = nx.pagerank(
            G,
            alpha=alpha,
            weight=weight,
            max_iter=max_iter,
        )
    except nx.PowerIterationFailedConvergence:
        print(
            "WARNING: PageRank did not converge. Returning NaN values "
            "for all graph nodes."
        )
        pagerank_values = {node: float("nan") for node in G.nodes}

    return pd.Series(pagerank_values, name="pagerank")


def compute_centrality_metrics(G, weight="weight"):
    return pd.concat(
        [
            compute_in_strength(G, weight),
            compute_out_strength(G, weight),
            compute_eigenvector_centrality(G, weight),
            compute_reverse_eigenvector_centrality(G, weight),
            compute_pagerank(G, weight),
        ],
        axis=1,
    )