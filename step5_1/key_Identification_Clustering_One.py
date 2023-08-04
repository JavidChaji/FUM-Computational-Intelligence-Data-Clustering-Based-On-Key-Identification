import gensim
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
import math

def save_function(original_cosine_distance, current_distance, indexes, document_cluster_labels, n_cluster_previous, n_cluster_current):
    np.save('step5_1/original_cosine_distance', original_cosine_distance)
    np.save('step5_1/indexes', indexes)
    np.save('step5_1/first_current_distance_of_vectors', current_distance)

def key_identification_clustering_one(document_vectors, n_clusters):

    original_cosine_distance = np.zeros(shape=(8695, 8695))

    for i in range(len(document_vectors.distances(document_vectors[0]))):
        #calc cosine distances
        original_cosine_distance[i] = document_vectors.distances(document_vectors[i])

    original_cosine_distance = np.where(original_cosine_distance < 0, 0, original_cosine_distance)
    
    #calc nearest neighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nearest_neighbors.fit(original_cosine_distance)

    #Fetch kneighbors
    distances, indexes = nearest_neighbors.kneighbors()
    document_cluster_labels = [x for x in range(1, len(original_cosine_distance)+1)]
    n_cluster_previous = len(Counter(document_cluster_labels).keys())
    n_cluster_current = math.floor(n_cluster_previous/g)
    current_distance = np.zeros(shape=(n_cluster_previous, n_cluster_previous))

    for i in range(n_cluster_previous):
        for j in range(n_cluster_previous):
            current_distance_sigma = 0
            for l in range(k):
                for p in range(k):
                    current_distance_sigma += original_cosine_distance[indexes[i][l]][indexes[j][p]]
            
            current_distance[i][j] = current_distance_sigma/((k+1)**2)
        print("% ", ((i)/(n_cluster_previous))*100)
    print(current_distance)

    save_function(original_cosine_distance, current_distance, indexes, document_cluster_labels, n_cluster_previous, n_cluster_current)
    return document_vectors

k = 3
g = 40
n_cluster_target = 3

model = gensim.models.doc2vec.Doc2Vec.load("step4/doc2vec.model")

key_identification_clustering_one(model.dv, n_cluster_target)
