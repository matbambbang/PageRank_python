import numpy as np
import scipy.sparse
import os
import time
import argparse

from preprocessing import preprocessing
from pagerank import PageRank, TopicSensitivePageRank

def main(args) :
    # Preprocessing Step
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

    pr_start = time.time()
    pr.converge()
    pr_end = time.time()
    print("Power iteration - {} required time: {:.3f}seconds".format(args.pagerank, pr_end-pr_start))

    pr_result = []
    for query_ID in data_dicts['search_relevance_score'].keys() :
        candidate_indices, retrieval_scores = data_dicts['search_relevance_score'][query_ID]
        user_topic_prob = data_dicts['user_topic_probs'][query_ID]
        query_topic_prob = data_dicts['query_topic_probs'][query_ID]

        if args.pagerank == "gpr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, criterion=args.criterion)
        elif args.pagerank == "ptspr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, user_topic_prob, criterion=args.criterion)
        elif args.pagerank == "qtspr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, query_topic_prob, criterion=args.criterion)

        for idx in range(len(candidate_indices)) :
            # Print function
            temp = [[]]
            temp[0].append(query_ID)
            temp[0].append("Q0")
            temp[0].append(str(pr_indices[idx] + 1))
            temp[0].append(str(idx + 1))
            temp[0].append(str(pr_scores[idx]))
            temp[0].append(args.cfg)
            pr_str = " ".join(temp[0])
            pr_result.append(pr_str)

    pr_result_text = "\n".join(pr_result)

    with open(args.pagerank + "_" + args.cfg + ".txt", "w") as f :
        f.write(pr_result_text)

    pr_end = time.time()
    print("total {} required time : {:.3f}seconds".format(args.pagerank, pr_end - pr_start))
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
    parser.add_argument("--pagerank", type=str, default="gpr", choices=["gpr", "ptspr", "qtspr"])
    parser.add_argument("--cfg", type=str, default="run-1")

    args = parser.parse_args()
    main(args)