import numpy as np
import scipy.sparse
import pickle
from overrides import overrides

from preprocessing import preprocessing

class PageRank(object) :
    def __init__(self, trans_matrix, dampening_factor=0.8, **kwargs) :
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        self.num_docs = trans_matrix.shape[0]
        self.matrix = trans_matrix.transpose()
        self.df = dampening_factor
        self.bias = np.repeat(1/self.num_docs, self.num_docs)
        self.interpolation()

    def interpolation(self):
        rowwise_sum = self.matrix.transpose().sum(axis=1).view(np.ndarray).squeeze() - 1
        self.inter_factor = -rowwise_sum / self.num_docs

    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias + self.df * np.dot(vector, self.inter_factor)
        return vector

    def converge(self, init_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        vector = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        # prev_vector = vector.copy()

        print("========== Power Iteration ==========")
        iter_no = 0
        while True :
            prev_vector = vector.copy()
            vector = self.iteration(vector)
            iter_no += 1
            difference = np.sum(np.abs(vector - prev_vector))
            if iter_no % 10 == 0 :
                print("Iter {0:4d} | Difference : {1:4.4f}".format(iter_no, difference))
            if difference < stop_criterion :
                print("Iter {0:4d} | Difference : {1:4.4f}".format(iter_no, difference))
                break

        self.ranked_vector = vector
        print("================ End ================")

    def scoring_function(self, candidate_index, retrieval_score, ranked_vector, criterion="ns") :
        available_criterion = ["ns", "ws", "cm"]
        assert criterion in available_criterion
        if criterion == "ns" :
            score = ranked_vector
        elif criterion == "ws" :
            retrieval_weight = 1.
            pagerank_weight = 0.
            score = retrieval_weight * retrieval_score + pagerank_weight * ranked_vector
        elif criterion == "cm" :
            score = np.zeros(ranked_vector.shape)

        score_ranking = np.argsort(score)[::-1]
        # print(score_ranking)
        # print(len(score_ranking))
        # print(len(candidate_index))
        # print(candidate_index)
        # print(type(score_ranking))
        # print(type(candidate_index))
        # sorted_index = candidate_index[score_ranking]
        sorted_index = np.array(candidate_index)[score_ranking]
        # print("======")
        # print(score.shape)
        # print(type(score))
        sorted_score = score[score_ranking]
        # print("===================================")
        return sorted_index, sorted_score

    def ranking(self, candidate_index, retrieval_score, criterion="ns", stop_criterion=None, pre_computed=True) :
        if not pre_computed :
            self.converge(stop_criterion=stop_criterion)
        pagerank = self.ranked_vector[candidate_index]
        ranking_result  = self.scoring_function(candidate_index, retrieval_score, pagerank, criterion=criterion)
        return ranking_result


class TopicSensitivePageRank(PageRank):
    def __init__(self, trans_matrix, topic_matrix, dampening_factor=0.8, topic_factor=0.1):
        super(TopicSensitivePageRank, self).__init__(trans_matrix, dampening_factor=dampening_factor)
        assert dampening_factor + topic_factor <= 1
        self.tf = topic_factor
        self.tmat = topic_matrix
        self.num_topics = topic_matrix.shape[1]
        self.bias = np.vstack([self.bias for _ in range(self.num_topics)]).transpose()

    @overrides
    def interpolation(self):
        rowwise_sum = self.matrix.transpose().sum(axis=1).view(np.ndarray).squeeze() - 1
        self.inter_factor = (-rowwise_sum / self.num_docs).reshape(1,-1)

    @overrides
    def iteration(self, matrix):
        matrix = self.df * self.matrix * matrix + self.tf * self.tmat + (1 - self.df - self.tf) * self.bias + self.df * np.dot(self.inter_factor,matrix)
        return matrix

    @overrides
    def converge(self, init_vector=None, topic_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        matrix = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        matrix = np.vstack([matrix for _ in range(self.num_topics)]).transpose()
        # prev_matrix = matrix.copy()

        print("========== Power Iteration ==========")
        iter_no = 0
        while True :
            prev_matrix = matrix.copy()
            matrix = self.iteration(matrix)
            iter_no += 1
            difference = np.sum(np.abs(matrix - prev_matrix)) / self.num_topics
            if iter_no % 10 == 0 :
                print("Iter {0:4d} | Difference : {1:4.4f}".format(iter_no, difference))
            if difference < stop_criterion :
                print("Iter {0:4d} | Difference : {1:4.4f}".format(iter_no, difference))
                break

        self.ranked_matrix = matrix
        print("================ End ================")
        return matrix

    @overrides
    def ranking(self, candidate_index, retrieval_score, topic_probs, criterion="ns", stop_criterion=None, pre_computed=True):
        # topic_probs shape : (12,1)
        if not pre_computed :
            self.converge(stop_criterion=stop_criterion)
        tpagerank = (self.ranked_matrix * topic_probs.reshape(12,1)).view(np.ndarray).squeeze()
        # print(tpagerank.shape)
        # print(type(tpagerank))
        # Shape error!!!!
        # print(tpagerank.shape)
        tpagerank = tpagerank[candidate_index]

        ranking_result = self.scoring_function(candidate_index, retrieval_score, tpagerank, criterion=criterion)
        return ranking_result

if __name__ == "__main__" :
    transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    doc_topic_matrix = scipy.sparse.load_npz("./data/doc_topic_matrix.npz")
    gpr = PageRank(trans_matrix=transition_matrix, dampening_factor=0.8)
    tspr = TopicSensitivePageRank(trans_matrix=transition_matrix, topic_matrix=doc_topic_matrix, dampening_factor=0.8, topic_factor=0.1)

    gpr.converge()
    print("Sample Ranked Vector : ", gpr.ranked_vector)
    print("GPR Convergence Checked!")
    tspr.converge()
    print("Sample Ranked Matrix : ", tspr.ranked_matrix)
    print("TSPR Convergence Checked!")