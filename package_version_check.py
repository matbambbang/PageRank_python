import numpy as np
import scipy
import scipy.sparse as sparse

print("Numpy Version Check")
print(np.__version__)
print("Scipy Version Check")
print(scipy.__version__)

row = [0,0,1,1,2]
col = [0,1,1,2,3]
data = [1,1,1,1,1]
mat = sparse.coo_matrix((data, (row,col)), shape=(4,4))
# print(mat.toarray())
# print(mat.sum())
# print(mat.sum(axis=0))

mat = mat.tocsr()
print(mat.toarray())
columnwise_sum = mat.sum(axis=1)
add_term = columnwise_sum.copy()
zero_term = add_term==0
add_term[zero_term] = -1
nonzero_term = add_term>0
add_term[nonzero_term] = 0
print(add_term)

print(type(mat))
add_term = sparse.csr_matrix(np.tile(add_term, 4))
mat = mat - add_term
print(type(mat))
columnwise_sum = mat.sum(axis=1)
print(mat.multiply(1/columnwise_sum).toarray())