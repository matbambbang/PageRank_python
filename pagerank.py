import numpy as np
import scipy.sparse
import pickle
from overrides import overrides

from preprocessing import preprocessing

class PageRank(object) :
    def __init__(self, trans_matrix, dampening_factor=0.8, bias_control=True, **kwargs) :
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        self.num_docs = trans_matrix.shape[0]
        # previously transpose the matrix
        self.matrix = trans_matrix.transpose()
        self.df = dampening_factor
        # self.bias = np.repeat(1/self.num_docs, self.num_docs)
        self.bias_setup(bias_control)

    def bias_setup(self, bias_control=True) :
        self.bias = np.repeat(1/self.num_docs, self.num_docs)
        if bias_control :
            self.bias_correction()

    def bias_correction(self) :
        rowwise_sum = self.matrix.sum(axis=1).view(np.ndarray).squeeze() - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
        # print("row sum", rowwise_sum.shape)
        # print(type(rowwise_sum))
        # print(self.bias.shape)
        self.bias -= 1/(1-self.df) * rowwise_sum

    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias
        return vector

    def converge(self, init_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        vector = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        prev_vector = vector.copy()

        iter_no = 0
        while True :
            prev_vector = vector.copy()
            vector = self.iteration(vector)
            iter_no += 1
            difference = np.sum(np.abs(vector - prev_vector))
            if iter_no % 100 == 0 :
                print("Iter {0:5d} | Difference : {1:4.4f}".format(iter_no, difference))
            if difference < stop_criterion :
                break

        self.ranked_vector = vector
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

# class PersonalizedPageRank(PageRank) :
#     def __init__(self, trans_matrix, personalized_vector, dampening_factor=0.8, personalized_factor=0.1) :
#         super(PersonalizedPageRank,self).__init__(trans_matrix, dampening_factor=dampening_factor, bias_control=False)
#         assert dampening_factor + personalized_factor <= 1
#         self.pf = personalized_factor
#         self.pvec = personalized_vector
#         self.bias_correction()
#
#     @overrides
#     def bias_correction(self) :
#         rowwise_sum = self.matrix.sum(axis=1).view(np.ndarray).squeeze() - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
#         self.bias -= (1-self.pf)/(1-self.df-self.pf) * rowwise_sum
#
#     @overrides
#     def iteration(self, vector) :
#         vector = self.df * self.matrix * vector + self.pf * self.pvec + (1-self.df-self.pf) * self.bias
#         return vector
#
# class QuerySensitivePageRank(PageRank) :
#     def __init__(self, trans_matrix, query_vector, dampening_factor=0.8, query_factor=0.1) :
#         super(QuerySensitivePageRank,self).__init__(trans_matrix, dampening_factor=dampening_factor, bias_control=False)
#         assert dampening_factor + query_factor <= 1
#         self.qf = query_factor
#         self.qvec = query_vector
#         self.bias_correction()
#
#     @overrides
#     def bias_correction(self) :
#         rowwise_sum = self.matrix.sum(axis=1).view(np.ndarray).squeeze() - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
#         self.bias -= (1-self.qf)/(1-self.df-self.qf) * rowwise_sum
#
#     @overrides
#     def iteration(self, vector) :
#         vector = self.df * self.matrix * vector + self.qf * self.qvec + (1-self.df-self.qf) * self.bias
#         return vector

class TopicSensitivePageRank(PageRank):
    def __init__(self, trans_matrix, topic_matrix, dampening_factor=0.8, topic_factor=0.1):
        super(TopicSensitivePageRank, self).__init__(trans_matrix, dampening_factor=dampening_factor, bias_control=False)
        assert dampening_factor + query_factor <= 1
        self.tf = topic_factor
        self.tmat = topic_matrix
        self.bias_correction()

    @overrides
    def bias_correction(self):
        rowwise_sum = self.matrix.sum(axis=1).view(np.ndarray).squeeze() - 1  # -1 does : sum=1 elem to 0, sum=0 elem to -1
        self.bias -= (1 - self.tf) / (1 - self.df - self.tf) * rowwise_sum

    @overrides
    def iteration(self, vector, topic_vector):
        vector = self.df * self.matrix * vector + self.tf * topic_vector + (1 - self.df - self.tf) * self.bias
        return vector

    @overrides
    def converge(self, init_vector=None, topic_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        vector = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        prev_vector = vector.copy()

        iter_no = 0
        while True :
            prev_vector = vector.copy()
            vector = self.iteration(vector, topic_vector)
            iter_no += 1
            difference = np.sum(np.abs(vector - prev_vector))
            if iter_no % 100 == 0 :
                print("Iter {0:5d} | Difference : {1:4.4f}".format(iter_no, difference))
            if difference < stop_criterion :
                break

        return vector

    @overrides
    def recommend(self, retrieval_score, stop_criterion=None, topic_vector=None, pre_computed=False):
        assert topic_vector != None
        topic_ranked_vectors = []
        for topic_idx in range(self.tmat.shape[0]) :
            topic_ranked_vector = self.converge(topic_vector=self.tmat[topic_idx])
            topic_ranked_vectors.append(topic_ranked_vector)

        score = self.scoring_function(retrieval_score, topic_ranked_vectors)
        return score

    @overrides
    def scoring_function(self, retrieval_score, topic_ranked_vectors, criterion="dot") :
        return None

if __name__ == "__main__" :
    transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    with open("./data/user_topic_vector_dict.pkl", "rb") as f :
        user_topic_vector_dict = pickle.load(f)
    with open("./data/query_topic_vector_dict.pkl", "rb") as f :
        query_topic_vector_dict = pickle.load(f)
    sample_user_vec = user_topic_vector_dict[6][4]
    sample_query_vec = query_topic_vector_dict[2][1]
    gpr = PageRank(trans_matrix=transition_matrix, dampening_factor=0.8)
    # qtspr = QuerySensitivePageRank(trans_matrix=transition_matrix, query_vector=None, dampening_factor=0.8, query_factor=0.1)
    # ptspr = PersonalizedPageRank(trans_matrix=transition_matrix, personalized_vector=None, dampening_factor=0.8, personalized_factor=0.1)
    print("PageRank, QuerySensitivePageRank, PersonalizedPageRank Checked!")