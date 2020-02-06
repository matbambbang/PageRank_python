import numpy as np
import scipy.sparse
import os
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

    num_row = max(row_index)
    num_col = max(col_index)
    num_docs = max(num_row, num_col) + 1
    print("# of Documents : {}".format(num_docs))
    transition_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_docs,num_docs)).tocsr()

    norm_factor = transition_matrix.sum(axis=1)
    nonzero_row = norm_factor.nonzero()[0]
    print("# of non-zero rows : ", len(nonzero_row))
    transition_matrix = transition_matrix.multiply(1/norm_factor)
    print("Transition Matrix Properly Normalized")

    print("Transition matrix (without 1/n) constructed")
    return transition_matrix, num_docs

def struct_doc_topic(text_raw_path, num_docs) :
    # doc_topics.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    topic_set = []
    doc_topic = [[] for _ in range(num_docs)]
    row_index = []
    col_index = []
    values = []
    for elem in parsed_text[:-1] :
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
    doc_topic_matrix = doc_topic_matrix.multiply(1/norm_factor).transpose()
    # print(doc_topic_matrix.shape)
    print("Successfully normalize document-topic matrix, Shape : {}".format(doc_topic_matrix.shape))
    return doc_topic, doc_topic_matrix

def struct_user_topic_interest(text_raw_path) :
    # user_topic_distro.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    user_topic_vector_dict = dict()
    for elem in parsed_text[:-1] :
        elem_parse = elem.split(" ")
        summary_vec = []
        for score_str in elem_parse[2:] :
            score_parse = score_str.split(":")
            summary_vec.append(float(score_parse[1]))
        user_topic_vector_dict[elem_parse[0] + "-" + elem_parse[1]] = np.array(summary_vec)
    print("Successfully crawl user-topic interest!")
    return user_topic_vector_dict

def struct_query_topic_dist(text_raw_path) :
    # queery_topic_distro.txt
    with open(text_raw_path, "r") as f :
        parsed_text = f.read().split("\n")
    query_topic_vector_dict = dict()
    for elem in parsed_text[:-1] :
        elem_parse = elem.split(" ")
        summary_vec = []
        for score_str in elem_parse[2:] :
            score_parse = score_str.split(":")
            summary_vec.append(float(score_parse[1]))
        query_topic_vector_dict[elem_parse[0] + "-" + elem_parse[1]] = np.array(summary_vec)
    print("Successfully crawl query-topic information!")
    return query_topic_vector_dict

def struct_search_relevance(text_raw_path, num_docs) :
    # indri_lists / queryID.results.txt
    # queryID=a_b (a : userID, b : b-th query)
    # Total number of candidate: 17,885
    files = os.listdir(text_raw_path)

    search_relevance_dict = dict()
    for file_name in files :
        path = os.path.join(text_raw_path, file_name)
        query_info = file_name.split(".")[0]
        with open(path, "r") as f :
            raw_file = f.read().split("\n")

        indice_list = []
        relevance_list = []
        for line in raw_file[:-1] :
            parsed = line.split(" ")
            indice_list.append(int(parsed[2]) - 1)
            relevance_list.append(float(parsed[4]))

        search_relevance_dict[query_info] = (indice_list, np.array(relevance_list))

    return search_relevance_dict

def preprocessing(transition_matrix_path="./data/transition.txt",
                  doc_topics_path = "./data/doc_topics.txt",
                  user_topic_path = "./data/user-topic-distro.txt",
                  query_topic_path = "./data/query-topic-distro.txt",
                  search_relevance_path="./data/indri-lists"
                  ) :

    transition_matrix, num_docs = struct_trmatrix(transition_matrix_path)
    doc_topic, doc_topic_matrix = struct_doc_topic(doc_topics_path, num_docs)
    user_topic_vector_dict = struct_user_topic_interest(user_topic_path)
    query_topic_vector_dict = struct_query_topic_dist(query_topic_path)
    search_relevance_dict = struct_search_relevance(search_relevance_path, num_docs)

    scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
    scipy.sparse.save_npz("./data/doc_topic_matrix.npz", doc_topic_matrix)
    with open("./data/user_topic_vector_dict.pkl", "wb") as f :
        pickle.dump(user_topic_vector_dict, f)
    with open("./data/query_topic_vector_dict.pkl", "wb") as f :
        pickle.dump(query_topic_vector_dict, f)
    with open("./data/search_relevance_dict.pkl", "wb") as f :
        pickle.dump(search_relevance_dict, f)

    for elem in list(search_relevance_dict.keys()) :
        assert elem in list(user_topic_vector_dict.keys())
        assert elem in list(query_topic_vector_dict.keys())
    print("All possible Query selected")

    print("Preprocessing End.")
    summary_dict = {
        "transition_matrix": transition_matrix,
        "doc_topic_matrix": doc_topic_matrix,
        "user_topic_probs": user_topic_vector_dict,
        "query_topic_probs": query_topic_vector_dict,
        "search_relevance_score": search_relevance_dict
    }
    return summary_dict

if __name__ == "__main__" :
    # Preprocessing check
    preprocessing()