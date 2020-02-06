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
        # gpr = PageRank(trans_matrix=data_dicts['transition_matrix'], dampening_factor=args.dampening_factor)
        pr = PageRank(trans_matrix=data_dicts['transition_matrix'], dampening_factor=args.dampening_factor)
    elif args.pagerank == "ptspr" or args.pagerank == "qtspr" :
        # tspr = TopicSensitivePageRank(trans_matrix=data_dicts['transition_matrix'],
        #                               topic_matrix=data_dicts['doc_topic_matrix'],
        #                               dampening_factor=args.dampening_factor,
        #                               topic_factor=args.topic_factor)
        pr = TopicSensitivePageRank(trans_matrix=data_dicts['transition_matrix'],
                                    topic_matrix=data_dicts['doc_topic_matrix'],
                                    dampening_factor=args.dampening_factor,
                                    topic_factor=args.topic_factor)

    # Power iteration
    # st = time.time()
    # gpr.converge()
    # gpr_end = time.time()
    # tspr.converge()
    # tspr_end = time.time()
    pr_start = time.time()
    pr.converge()
    pr_end = time.time()
    # print("GPR required time: {:.3f}seconds".format(gpr_end - st))
    # print("TSPR required time: {:.3f}seconds".format(tspr_end - gpr_end))
    print("Power iteration - {} required time: {:.3f}seconds".format(args.pagerank, pr_end-pr_start))

    # gpr_result = []
    # ptspr_result = []
    # qtspr_result = []
    pr_result = []
    for query_ID in data_dicts['search_relevance_score'].keys() :
        candidate_indices, retrieval_scores = data_dicts['search_relevance_score'][query_ID]
        user_topic_prob = data_dicts['user_topic_probs'][query_ID]
        query_topic_prob = data_dicts['query_topic_probs'][query_ID]

        # gpr_indices, gpr_scores = gpr.ranking(candidate_indices, retrieval_scores, criterion=args.criterion)
        # ptspr_indices, ptspr_scores = tspr.ranking(candidate_indices, retrieval_scores, user_topic_prob, criterion=args.criterion)
        # qtspr_indices, qtspr_scores = tspr.ranking(candidate_indices, retrieval_scores, query_topic_prob, criterion=args.criterion)
        if args.pagerank == "gpr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, criterion=args.criterion)
        elif args.pagerank == "ptspr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, user_topic_prob, criterion=args.criterion)
        elif args.pagerank == "qtspr" :
            pr_indices, pr_scores = pr.ranking(candidate_indices, retrieval_scores, query_topic_prob, criterion=args.criterion)

        for idx in range(len(candidate_indices)) :
            # Print function
            # temp = [[],[],[]]
            temp = [[]]
            temp[0].append(query_ID)
            # temp[1].append(query_ID)
            # temp[2].append(query_ID)

            temp[0].append("Q0")
            # temp[1].append("Q0")
            # temp[2].append("Q0")

            # temp[0].append(str(gpr_indices[idx]+1))
            # temp[1].append(str(ptspr_indices[idx]+1))
            # temp[2].append(str(qtspr_indices[idx]+1))
            temp[0].append(str(pr_indices[idx] + 1))

            temp[0].append(str(idx + 1))
            # temp[1].append(str(idx + 1))
            # temp[2].append(str(idx + 1))

            # temp[0].append(str(gpr_scores[idx]))
            # temp[1].append(str(ptspr_scores[idx]))
            # temp[2].append(str(qtspr_scores[idx]))
            temp[0].append(str(pr_scores[idx]))

            temp[0].append(args.cfg)
            # temp[1].append("run-1")
            # temp[2].append("run-1")

            # gpr_str = " ".join(temp[0])
            # ptspr_str = " ".join(temp[1])
            # qtspr_str = " ".join(temp[2])
            pr_str = " ".join(temp[0])

            # gpr_result.append(gpr_str)
            # ptspr_result.append(ptspr_str)
            # qtspr_result.append(qtspr_str)
            pr_result.append(pr_str)

    # gpr_result_text = "\n".join(gpr_result)
    # ptspr_result_text = "\n".join(ptspr_result)
    # qtspr_result_text = "\n".join(qtspr_result)
    pr_result_text = "\n".join(pr_result)

    # Save results
    # with open("gpr_"+args.cfg+".txt", "w") as f :
    #     f.write(gpr_result_text)
    # with open("ptspr_"+args.cfg+".txt", "w") as f :
    #     f.write(ptspr_result_text)
    # with open("qtspr_"+args.cfg+".txt", "w") as f :
    #     f.write(qtspr_result_text)
    with open(args.pagerank + "_" + args.cfg + ".txt", "w") as f :
        f.write(pr_result_text)

    pr_end = time.time()
    print("total {} required time : {:.3f}".format(args.pagerank, pr_end - pr_start))
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