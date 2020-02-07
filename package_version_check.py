import numpy as np
import scipy
import os
import time
import argparse

from preprocessing import preprocessing
from pagerank import PageRank, TopicSensitivePageRank

def sample_generation(args) :
    # Preprocessing Step
    print("Numpy Version Check")
    print(np.__version__)
    print("Scipy Version Check")
    print(scipy.__version__)
    data_dicts = preprocessing(transition_matrix_path=args.transition_matrix,
                               doc_topics_path=args.document_topic,
                               user_topic_path=args.user_topic_interest,
                               query_topic_path=args.query_topic_relation,
                               search_relevance_path=args.search_relevance)

    # GPR, PTSPR, QTSPR construction
    if args.pagerank == "gpr" :
        pr = PageRank(trans_matrix=data_dicts['transition_matrix'], dampening_factor=args.dampening_factor)
    elif args.pagerank == "ptspr" or args.pagerank == "qtspr" :
        pr = TopicSensitivePageRank(trans_matrix=data_dicts['transition_matrix'],
                                    topic_matrix=data_dicts['doc_topic_matrix'],
                                    dampening_factor=args.dampening_factor,
                                    topic_factor=args.topic_factor)

    pr.converge()

    if args.pagerank == "gpr" :
        np.savetxt("GPR.txt", pr.ranked_vector, delimiter=" ")
    elif args.pagerank == "ptspr" :
        topic_prob = data_dicts['user_topic_probs']["2-1"]
        vector = (pr.ranked_matrix * topic_prob.reshape(12,1)).view(np.ndarray).squeeze()
        np.savetxt("QTSPR-U2Q1.txt", vector, delimiter=" ")
    elif args.pagerank == "qtspr" :
        topic_prob = data_dicts['query_topic_probs']["2-1"]
        vector = (pr.ranked_matrix * topic_prob.reshape(12,1)).view(np.ndarray).squeeze()
        np.savetxt("PTSPR-U2Q1.txt", vector, delimiter=" ")
    print("===================== END =====================")

if __name__ == "__main__" :
    parser = argparse.ArgumentParser("PageRank")
    parser.add_argument("--transition_matrix", type=str, default="./data/transition.txt")
    parser.add_argument("--document_topic", type=str, default="./data/doc_topics.txt")
    parser.add_argument("--user_topic_interest", type=str, default="./data/user-topic-distro.txt")
    parser.add_argument("--query_topic_relation", type=str, default="./data/query-topic-distro.txt")
    parser.add_argument("--search_relevance", type=str, default="./data/indri-lists")
    parser.add_argument("--dampening_factor", type=float, default=0.8)
    parser.add_argument("--topic_factor", type=float, default=0.1)
    parser.add_argument("--criterion", type=str, default="ws", choices=["ns", "ws", "cm"])
    parser.add_argument("--pagerank", type=str, default="qtspr", choices=["gpr", "ptspr", "qtspr"])
    parser.add_argument("--cfg", type=str, default="run-1")

    args = parser.parse_args()
    sample_generation(args)