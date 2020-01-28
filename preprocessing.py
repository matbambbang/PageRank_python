import numpy as np
# from scipy.io import mmread
import scipy.sparse
import pickle

def raw_check(transition_matrix_raw) :
    print("Transition Matrix Check")
    print("# of elements of Transition Matrix : ", len(transition_matrix_raw))
    print("Head Information : ")
    for i  in range(3) :
        print(transition_matrix_raw[i])
    print("Successfully read 'transition.txt'!")

def struct_trmatrix(text_raw_path) :
    # This function preprocess the transition.txt to scipy.csr_matrix format
    print("Constructing Transition Matrix")
    with open(text_raw_path, "r") as f :
        text_raw = f.read().split("\n")

    raw_check(text_raw)
    row_index = []
    col_index = []
    values = []
    for elem in text_raw[:-1] :
        elem_parse = elem.split(" ")
        # Python uses 0-start indexing, so apply -1 to each row, column index
        row_index.append(int(elem_parse[0])-1)
        col_index.append(int(elem_parse[1])-1)
        values.append(float(elem_parse[2]))

    # If I should make the matrix "symmetric"? # maybe, NO
    # row_copy = row_index.copy()
    # row_index.extend(col_index)
    # col_index.extend(row_copy)
    # values.extend(values)
    # num_docs = max(row_index) + 1
    num_row = max(row_index)
    num_col = max(col_index)
    num_docs = max(num_row, num_col) + 1
    print("# of Documents : {}".format(num_docs))
    transition_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_docs,num_docs)).tocsr()
    # print("hello")
    # with open(text_raw_path, "r") as f :
    #     transition_matrix = mmread(text_raw_path)
    # transition_matrix = mmread(text_raw_path)
    # print(transition_matrix.shape)
    # exit()
    #### Preprocessing #### ????
    # matrix.sum(axis=0) : sum all rows, shape: (row,1)
    # matrix.sum(axis=1) : sum all columns, shape: (1,col)
    # sum_elem = transition_matrix.sum(axis=1)
    # nonzero_row = sum_elem.nonzero()[0]
    # zero_idx = sum_elem==0
    # sum_elem[zero_idx] = -1
    # nonzero_idx = sum_elem>0
    # sum_elem[nonzero_idx] = 0
    # sum_elem = scipy.sparse.hstack([scipy.sparse.csr_matrix(sum_elem) for _ in range(num_docs)])
    # print("# of non-zero rows : ", len(nonzero_row))  # of nonzero column : 76586 (non-symmetric ver.)
    # for i in range(num_docs) :
    #     if i not in nonzero_row :
    #         transition_matrix[i] = 1
    norm_factor = transition_matrix.sum(axis=1)
    nonzero_row = norm_factor.nonzero()[0]
    print("# of non-zero rows : ", len(nonzero_row))
    transition_matrix = transition_matrix.multiply(1/norm_factor)
    print("Transition Matrix Properly Normalized")
    # transition_matrix = transition_matrix.tolil()
    zero_eliminate_fail = True
    if zero_eliminate_fail :
        print("="*80)
        print("QUESTION")
        print("="*80)
        print("I failed to eliminate zero rows :(")
        print("Anyway, not-perfect Transition matrix constructed")
        # scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
        return transition_matrix, num_docs
    transition_matrix = transition_matrix.todok()
    # for i in range(num_docs) :
    #     if i not in nonzero_row :
    #         transition_matrix[i] = 1/num_docs
    zero_row = [elem for elem in range(num_docs) if elem not in nonzero_row]
    for elem in zero_row :
        transition_matrix[elem] = 1/num_docs
    transition_matrix = transition_matrix.tocsr()
    # scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
    # transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    print("Successfully Saved to './data/transition_matrix.npz'!")
    return transition_matrix, num_docs

def struct_doc_topic(text_raw_path, num_docs) :
    # doc_topics.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    # print(parsed_text[-2])
    topic_set = []
    doc_topic = [[] for _ in range(num_docs)]
    row_index = []
    col_index = []
    values = []
    for elem in parsed_text[:-1] :
        # print("hello")
        elem_parse = elem.split(" ")
        topic_set.append(int(elem_parse[1]))
        doc_topic[int(elem_parse[0])-1].append(int(elem_parse[1]))
        row_index.append(int(elem_parse[0])-1)
        col_index.append(int(elem_parse[1])-1)
        values.append(1)
    topic_set = list(set(topic_set))
    print("Available Topics : ", topic_set)

    doc_topic_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_docs,len(topic_set))).tocsr().transpose()
    norm_factor = doc_topic_matrix.sum(axis=1)
    # doc_topic_matrix : (#_topic, #_doc)
    doc_topic_matrix = doc_topic_matrix.multiply(1/norm_factor)
    # print(doc_topic_matrix.shape)
    print("Successfully normalize document-topic matrix")
    return doc_topic, doc_topic_matrix

def struct_user_topic_interest(text_raw_path) :
    # user_topic_distro.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    # print(len(parsed_text))
    user_topic_dict = dict()
    user_topic_vector_dict = dict()
    for elem in parsed_text[:-1] :
        elem_parse = elem.split(" ")
        # need to convert topic score to vector?
        summary = dict()
        summary_vec = []
        summary["query"] = int(elem_parse[1])
        for score_str in elem_parse[2:] :
            # maybe, each score must describe all the scores related to topics
            # print(score_str)
            score_parse = score_str.split(":")
            # print(score_parse)
            summary[int(score_parse[0])] = float(score_parse[1])
            summary_vec.append(float(score_parse[1]))
        if int(elem_parse[0]) not in user_topic_dict.keys() :
            user_topic_dict[int(elem_parse[0])] = {}
            user_topic_vector_dict[int(elem_parse[0])] = {}
        user_topic_dict[int(elem_parse[0])][int(elem_parse[1])] = summary
        user_topic_vector_dict[int(elem_parse[0])][int(elem_parse[1])] = np.array(summary_vec)
        # user_topic_dict[int(elem_parse[0])] = summary
        # user_topic_vector_dict[int(elem_parse[0])] = {"query": int(elem_parse[1]), "vec": np.array(summary_vec)}

    print("Successfully crawl user-topic interest!")
    return user_topic_dict, user_topic_vector_dict

def struct_query_topic_dist(text_raw_path) :
    # queery_topic_distro.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    # print(len(parsed_text))
    query_topic_dict = dict()
    query_topic_vector_dict = dict()
    for elem in parsed_text[:-1] :
        elem_parse = elem.split(" ")
        # need to convert topic score to vector?
        summary = dict()
        summary_vec = []
        summary["query"] = int(elem_parse[1])
        for score_str in elem_parse[2:] :
            # maybe, each score must describe all the scores related to topics
            # print(score_str)
            score_parse = score_str.split(":")
            # print(score_parse)
            summary[int(score_parse[0])] = float(score_parse[1])
            summary_vec.append(float(score_parse[1]))
        if int(elem_parse[0]) not in query_topic_dict.keys() :
            query_topic_dict[int(elem_parse[0])] = {}
            query_topic_vector_dict[int(elem_parse[0])] = {}
        query_topic_dict[int(elem_parse[0])][int(elem_parse[1])] = summary
        query_topic_vector_dict[int(elem_parse[0])][int(elem_parse[1])] = np.array(summary_vec)
        # query_topic_dict[int(elem_parse[0])] = summary
        # query_topic_vector_dict[int(elem_parse[0])] = {"query": int(elem_parse[1]), "vec": np.array(summary_vec)}

    print("Successfully crawl query-topic information!")
    return query_topic_dict, query_topic_vector_dict

def struct_search_relevance(text_raw_path) :
    # indri_lists / queryID.results.txt
    # queryID=a_b (a : userID, b : b-th query)
    return None

def preprocessing(transition_matrix_path="./data/transition.txt",
                  doc_topics_path = "./data/doc_topics.txt",
                  user_topic_path = "./data/user-topic-distro.txt",
                  query_topic_path = "./data/query-topic-distro.txt"
                  ) :

    transition_matrix, num_docs = struct_trmatrix(transition_matrix_path)
    doc_topic, doc_topic_matrix = struct_doc_topic(doc_topics_path, num_docs)
    user_topic_dict, user_topic_vector_dict = struct_user_topic_interest(user_topic_path)
    query_topic_dict, query_topic_vector_dict = struct_query_topic_dist(query_topic_path)

    scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
    scipy.sparse.save_npz("./data/doc_topic_matrix.npz", doc_topic_matrix)
    with open("./data/user_topic_dict.pkl", "wb") as f :
        pickle.dump(user_topic_dict, f)
    with open("./data/user_topic_vector_dict.pkl", "wb") as f :
        pickle.dump(user_topic_vector_dict, f)
    with open("./data/query_topic_dict.pkl", "wb") as f :
        pickle.dump(query_topic_dict, f)
    with open("./data/query_topic_vector_dict.pkl", "wb") as f :
        pickle.dump(query_topic_vector_dict, f)

    print("Preprocessing End.")

if __name__ == "__main__" :
    preprocessing()
