import numpy as np
import scipy.sparse

def raw_check(transition_matrix_raw) :
    print("Transition Matrix Check")
    print("# of elements of Transition Matrix : ", len(transition_matrix_raw))
    print("Head")
    for i  in range(5) :
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
        row_index.append(int(elem_parse[0]))
        col_index.append(int(elem_parse[1]))
        values.append(float(elem_parse[2]))

    num_row = max(row_index)
    num_col = max(col_index)
    print("Maximum Row : {}, Column : {}".format(num_row, num_col))
    transition_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_row+1,num_col+1))
    # transition_matrix = scipy.sparse.coo_matrix((values, (row_index, col_index)), shape=(num_row + 1, num_col + 1))
    scipy.sparse.save_npz("./data/transition_matrix.npz", transition_matrix)
    # transition_matrix = scipy.sparse.load_npz("./data/transition_matrix.npz")
    print("Successfully Saved to './data/transition_matrix.npz'!")
    return transition_matrix

def main() :
    text_raw_path = "./data/transition.txt"
    preprocessing(text_raw_path)

if __name__ == "__main__" :
    main()