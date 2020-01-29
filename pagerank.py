import numpy as np
import scipy.sparse
import pickle
from overrides import overrides

from preprocessing import preprocessing

class PageRank(object) :
    def __init__(self, trans_matrix, dampening_factor=0.8, **kwargs) :
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        self.num_docs = trans_matrix.shape[0]
        # previously transpose the matrix
        self.matrix = trans_matrix.transpose()
        self.df = dampening_factor
        self.bias = np.repeat(1/self.num_docs, self.num_docs)
        # self.bias_setup(bias_control)
        self.interpolation()

    # def bias_setup(self, bias_control=True) :
    #     self.bias = np.repeat(1/self.num_docs, self.num_docs)
    #     if bias_control :
    #         self.bias_correction()
    #
    # def bias_correction(self) :
    #     rowwise_sum = self.matrix.sum(axis=1).view(np.ndarray).squeeze() - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
    #     print("row sum", rowwise_sum.shape)
    #     print(type(rowwise_sum))
    #     print(self.bias.shape)
    #     self.bias -= 1/(1-self.df) * rowwise_sum

    def interpolation(self):
        rowwise_sum = self.matrix.transpose().sum(axis=1).view(np.ndarray).squeeze() - 1
        self.inter_factor = -rowwise_sum / self.num_docs

    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias + self.df * np.dot(vector, self.inter_factor)
        return vector

    def converge(self, init_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        vector = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        prev_vector = vector.copy()

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
        return vector

    def scoring_function(self, retrieval_score, ranked_vector, criterion="dot") :
        if criterion == "dot" :
            return retrieval_score * ranked_vector
        return None

    def recommend(self, retrieval_score, stop_criterion=None, pre_computed=False) :
        if not pre_computed :
            ranked_vector = self.converge(stop_criterion=stop_criterion)
        # retrieval_score = np.dot(doc_vector, query_vector)
        score = self.scoring_function(retrieval_score, self.ranked_vector)
        return score

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
        # print(type(self.inter_factor))
        # print(self.inter_factor.shape)

    @overrides
    def iteration(self, matrix):
        matrix = self.df * self.matrix * matrix + self.tf * self.tmat + (1 - self.df - self.tf) * self.bias + self.df * np.dot(self.inter_factor,matrix)
        # matrix += np.dot(self.inte)
        return matrix

    @overrides
    def converge(self, init_vector=None, topic_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        matrix = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        matrix = np.vstack([matrix for _ in range(self.num_topics)]).transpose()
        prev_matrix = matrix.copy()

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
    def recommend(self, retrieval_score, stop_criterion=None, topic_probs=None, pre_computed=False):
        # topic_probs shape : (12,1)
        assert topic_probs != None
        if not pre_computed :
            ranked_matrix = self.converge(stop_criterion=stop_criterion)
        else :
            ranked_matrix = self.ranked_matrix
        ranked_vector = ranked_matrix * topic_probs

        score = self.scoring_function(retrieval_score, ranked_vector)
        return score

if __name__ == "__main__" :
    transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    # with open("./data/user_topic_vector_dict.pkl", "rb") as f :
    #     user_topic_vector_dict = pickle.load(f)
    # with open("./data/query_topic_vector_dict.pkl", "rb") as f :
    #     query_topic_vector_dict = pickle.load(f)
    doc_topic_matrix = scipy.sparse.load_npz("./data/doc_topic_matrix.npz")
    # sample_user_vec = user_topic_vector_dict[6][4]
    # sample_query_vec = query_topic_vector_dict[2][1]
    gpr = PageRank(trans_matrix=transition_matrix, dampening_factor=0.8)
    tspr = TopicSensitivePageRank(trans_matrix=transition_matrix, topic_matrix=doc_topic_matrix, dampening_factor=0.8, topic_factor=0.1)
    print("PageRank Checked!")
    gpr.converge()
    # print(gpr.ranked_vector)
    # print(sum(gpr.ranked_vector))
    print("GPR Convergence Checked!")
    tspr.converge()
    print("Sample Ranked Matrix : ", tspr.ranked_matrix)
    # print(tspr.ranked_matrix.shape)
    # print(tspr.ranked_matrix.sum(axis=0))
    print("TSPR Convergence Checked!")
    # print(transition_matrix.sum(axis=1))