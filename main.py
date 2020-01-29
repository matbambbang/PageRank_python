import numpy as np
import scipy.sparse
import os
import argparse

from preprocessing import preprocessing
from pagerank import PageRank, TopicSensitivePageRank

def main() :
    return None

if __name__ == "__main__" :
    parser = argparse.ArgumentParser("PageRank")
    parser.add_argument("--transition_matrix", type=str, default="./data/transition.txt")
    parser.add_argument("--document_topic", type=str, default="./data/doc_topics.txt")
    parser.add_argument("--user_topic_interest", type=str, default="./data/user-topic-distro.txt")
    parser.add_argument("--query_topic_relation", type=str, default="./data/query-topic-distro.txt")
    parser.add_argument("--search_relevance", type=str, default="./data/indri-lists")
    parser.add_argument("--dampening_factor", type=float, default=0.8)
    parser.add_argument("--topic_factor", type=float, default=0.1)

    args = parser.parse_args()
    processed_result = preprocessing(transition_matrix_path=args.transition_matrix,
                                     doc_topics_path=args.document_topic,
                                     user_topic_path=args.user_topic_interest,
                                     query_topic_path=args.query_topic_relation,
                                     search_relevance_path=args.search_relevance
                                     )
    gpr = PageRank(trans_matrix=processed_result['transition_matrix'], dampening_factor=args.dampening_factor)
    ptspr = TopicSensitivePageRank(trans_matrix=processed_result['transition_matrix'],
                                   topic_factor=processed_result['doc_topic_matrix'],
                                   dampening_factor=args.dampening_factor,
                                   topic_factor=args.topic_factor)
    qtspr = TopicSensitivePageRank(trans_matrix=processed_result['transition_matrix'],
                                   topic_factor=processed_result['doc_topic_matrix'],
                                   dampening_factor=args.dampening_factor,
                                   topic_factor=args.topic_factor)
    return None