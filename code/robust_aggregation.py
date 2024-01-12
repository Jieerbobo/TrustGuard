import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


def ro_coefficient(node_embedding, prune, row, col):
    """
    Similar with GNNGuard: https://arxiv.org/abs/2006.08149
    To do: Speed up calculations.
    """
    node_embedding_copy = node_embedding.cpu().data.numpy()
    sim_matrix = cosine_similarity(node_embedding_copy)
    # print('sim_matrix',sim_matrix.max(),sim_matrix.min())  # Sim(X, Y) = <X, Y> / (||X||*||Y||)
    sim_matrix = 1 + sim_matrix  # range: [0,2]！！！
    sim = sim_matrix[row, col]
    sim[sim < prune] = 0
    # print('testing',sim.max(),sim.min())
    loop_list = [row[i] for i in range(len(row)) if row[i]==col[i]]

    att_dense = lil_matrix((node_embedding_copy.shape[0], node_embedding_copy.shape[0]), dtype=np.float32)
    att_dense[row, col] = sim
    att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format='lil')
    att_dense_norm = normalize(att_dense, axis=1, norm='l1')

    degree = (att_dense_norm != 0).sum(1)
    att_dense_tran = lil_matrix(att_dense_norm.A * degree.A)
    ones = np.zeros((len(degree),))
    for i in range(len(loop_list)):
        ones[i] = 1
    self_loop = sp.diags(ones, offsets=0, format='lil')
    # https://blog.csdn.net/mercies/article/details/108513787
    att_final = att_dense_tran + self_loop
    att_final_norm = normalize(att_final, axis=1, norm='l1')

    att_real = att_final_norm[row, col]
    att_real_final = np.reshape(att_real.A, (len(row), 1))
    att_real_final = torch.from_numpy(att_real_final).type(torch.float)

    return att_real_final
