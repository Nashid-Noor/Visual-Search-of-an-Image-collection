
import numpy as np 
from scipy.spatial.distance import euclidean,cityblock, cosine, hamming,mahalanobis


def cvpr_compare(F1, F2,metric):
    if(metric=='mahalanobis'):
        cov_matrix = np.cov(F2.T)  
        inv_cov_matrix = np.linalg.inv(cov_matrix)  
        distances = [
            mahalanobis(F1, desc, inv_cov_matrix)
            for desc in F2
        ]
        return distances

    max_length = max(len(F1), len(F2))
    F1 = np.pad(F1, (0, max_length - len(F1)), 'constant')
    F2 = np.pad(F2, (0, max_length - len(F2)), 'constant')

    if(metric=='euclidean'):  
        return euclidean(F1,F2)       
    elif metric == 'L1':
        return cityblock(F1, F2)
    elif metric == 'cosine':
        return cosine(F1, F2)
    elif metric =='hamming':
        return hamming(F1,F2)


        


   


