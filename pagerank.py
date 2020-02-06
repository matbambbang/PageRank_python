import numpy as np
import scipy.sparse
from overrides import overrides

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
            pagerank_weight = 1.
            retrieval_score /= np.sqrt(sum(retrieval_score ** 2))
            ranked_vector /= np.sqrt(sum(ranked_vector ** 2))
            score = retrieval_weight * retrieval_score + pagerank_weight * ranked_vector
        elif criterion == "cm" :
            # score = np.zeros(ranked_vector.shape)
            # score = np.random.randn(*ranked_vector.shape)
            retrieval_score -= np.mean(retrieval_score)
            ranked_vector -= np.mean(ranked_vector)
            score = np.tanh(retrieval_score) + np.tanh(ranked_vector)

        score_ranking = np.argsort(score)[::-1]
        if candidate_index is None :
            candidate_index = [i for i in range(len(ranked_vector))]
        sorted_index = np.array(candidate_index)[score_ranking]
        sorted_score = score[score_ranking]
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
        # print(matrix.sum(axis=0))
        print("================ End ================")
        return matrix

    @overrides
    def ranking(self, candidate_index, retrieval_score, topic_probs, criterion="ns", stop_criterion=None, pre_computed=True):
        # topic_probs shape : (12,1)
        if not pre_computed :
            self.converge(stop_criterion=stop_criterion)
        tpagerank = (self.ranked_matrix * topic_probs.reshape(12,1)).view(np.ndarray).squeeze()
        tpagerank = tpagerank[candidate_index]

        ranking_result = self.scoring_function(candidate_index, retrieval_score, tpagerank, criterion=criterion)
        return ranking_result

if __name__ == "__main__" :
    # Test code for evaluating PageRank
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