import numpy as np
import scipy.sparse

from transition_matrix import preprocessing

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
        self.bias = np.repeat(1/self.num_docs, num_docs)
        if bias_control :
            self.bias_correction()

    def bias_correction(self) :
        rowwise_sum = self.matrix.sum(axis=1) - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
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

        return vector

class PersonalizedPageRank(PageRank) :
    def __init__(self, trans_matrix, personalized_vector, dampening_factor=0.8, personalized_factor=0.1) :
        super(PersonalizedPageRank,self).__init__(trans_matrix, dampening_factor=dampening_factor, bias_control=False)
        assert dampening_factor + personalized_factor <= 1
        self.pf = personalized_factor
        self.pvec = personalized_vector
        self.bias_correction()

    @overrides
    def bias_correction(self) :
        rowwise_sum = self.matrix.sum(axis=1) - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
        self.bias -= (1-self.pf)/(1-self.df-self.pf) * rowwise_sum

    @overrides
    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + self.pf * self.pvec + (1-self.df-self.pf) * self.bias
        return vector

class QuerySensitivePageRank(PageRank) :
    def __init__(self, trans_matrix, query_vector, dampening_factor=0.8, query_factor=0.1) :
        super(QuerySensitivePageRank,self).__init__(trans_matrix, dampening_factor=dampening_factor, bias_control=False)
        self.qf = query_factor
        self.qvec = query_vector
        self.bias_correction()

    @overrides
    def bias_correction(self) :
        rowwise_sum = self.matrix.sum(axis=1) - 1 # -1 does : sum=1 elem to 0, sum=0 elem to -1
        self.bias -= (1-self.qf)/(1-self.df-self.qf) * rowwise_sum

    @overrides
    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + self.qf * self.qvec + (1-self.df-self.qf) * self.bias
        return vector

if __name__ == "__main__" :
    transition_matrix = scipy.sparse.load_npz("./data/transitino_matrix.npz")
    gpr = PageRank(trans_matrix=transition_matrix, dampening_factor=0.8)