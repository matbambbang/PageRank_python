import numpy as np
import scipy.sparse

from transition_matrix import preprocessing

class PageRank(object) :
    def __init__(self, trans_matrix, dampening_factor=0.8, **kwargs) :
        assert trans_matrix.shape[0] == trans_matrix.shape[1]
        self.num_docs = trans_matrix.shape[0]
        # previously transpose the matrix
        self.matrix = trans_matrix.transpose()
        self.df = dampening_factor
        self.bias = np.repeat(1/self.num_docs, self.num_docs)

    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias
        return vector

    def converge(self, init_vector=None, stop_criterion=None) :
        stop_criterion = stop_criterion or 1e-8
        vector = init_vector or np.repeat(1.0/self.num_docs, self.num_docs)
        prev_vector = vector.copy()

        while True :
            prev_vector = vector.copy()
            vector = self.iteration(vector)
            if np.sum(np.abs(vector - prev_vector)) < stop_criterion :
                break

        return vector

class PersonalizedPageRank(PageRank) :
    def __init__(self, trans_matrix, personalized_factor) :
        super(PersonalizedPageRank,self).__init__(trans_matrix)
        self.personalized = personalized_factor

    @overrides
    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias
        return vector

class QuerySensitivePageRank(PageRank) :
    def __init__(self, trans_matrix, query_factor) :
        super(QuerySensitivePageRank,self).__init__(trans_matrix)
        self.query = query_factor

    @overrides
    def iteration(self, vector) :
        vector = self.df * self.matrix * vector + (1-self.df) * self.bias
        return vector

if __name__ == "__main__" :
    def main()