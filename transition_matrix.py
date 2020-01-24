import numpy as np
import scipy.sparse

def raw_check(transition_matrix_raw) :
    print("Transition Matrix Check")
    print("# of elements of Transition Matrix : ", len(transition_matrix_raw))
    print("Head")
    for i  in range(3) :
        print(transition_matrix_raw[i])
    print("Successfully read files!")

def preprocessing(text_raw_path) :
    # This function preprocess the transition.txt to scipy.coo_matrix format
    print("Start Preprocessing")
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
    transition_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_docs,num_docs))
    #### Preprocessing ####
    # matrix.sum(axis=0) : sum all rows, shape: (row,1)
    # matrix.sum(axis=1) : sum all columns, shape: (1,col)
    sum_elem = transition_matrix.sum(axis=1)
    zero_idx = sum_elem==0
    sum_elem[zero_idx] = -1
    nonzero_idx = sum_elem>0
    sum_elem[nonzero_idx] = 0
    # sum_elem = scipy.sparse.csr_matrix(np.tile(sum_elem, num_docs))
    sum_elem = scipy.sparse.hstack([scipy.sparse.csr_matrix(sum_elem) for _ in range(num_docs)])
    print("Sum finished")
    transition_matrix -= sum_elem
    print("Filled")
    norm_factor = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix.multiply(1/norm_factor)
    print(type(transition_matrix))
    scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
    # transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    print("Successfully Saved to './data/transition_matrix.npz'!")
    return transition_matrix

def main() :
    text_raw_path = "./data/transition.txt"
    preprocessing(text_raw_path)

if __name__ == "__main__" :
    main()