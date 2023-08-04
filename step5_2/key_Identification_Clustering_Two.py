from xmlrpc.client import MAXINT
import gensim
from pyparsing import str_type
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from collections import Counter
from more_itertools import locate
import math
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import seaborn as sns
from numpy import array
import matplotlib.pyplot as plt

def load_function():
    original_cosine_distance = np.load('step5_1/original_cosine_distance.npy')
    knn_indexes = np.load('step5_1/indexes.npy')
    document_cluster_labels = [x for x in range(len(original_cosine_distance))]
    n_cluster_previous = len(Counter(document_cluster_labels).keys())
    n_cluster_current = math.floor(n_cluster_previous/g)
    current_distance = np.load('step5_1/first_current_distance_of_vectors.npy')
    return original_cosine_distance, current_distance, knn_indexes, document_cluster_labels, n_cluster_previous, n_cluster_current

def key_item_selection(current_distance_matrix, n_clusters):
    
    mean_distance_matrix = np.mean(current_distance_matrix, axis=1)
    
    #s in article
    first_selected_key = np.argmin(mean_distance_matrix)
    selected_keys = []
    selected_keys.append(first_selected_key)
    
    #k in article
    unselected_items_for_key = [x for x in range(0, len(current_distance_matrix)) if x != first_selected_key]
    temp_min_key_j = -1
    temp_max_key_i = -1
    n = len(selected_keys)
    
    while n < n_clusters:
        temp_max_distance = 0
        for i in unselected_items_for_key:
            temp_min_distance = MAXINT
            for j in selected_keys:
                if current_distance_matrix[i][j] < temp_min_distance:
                    temp_min_distance = current_distance_matrix[i][j]
                    temp_min_key_j = j
            if current_distance_matrix[i][temp_min_key_j] > temp_max_distance:
                temp_max_distance = current_distance_matrix[i][temp_min_key_j]
                temp_max_key_i = i
        selected_keys.append(temp_max_key_i)
        unselected_items_for_key.remove(temp_max_key_i)
        n = n + 1
    print(len(selected_keys))
    print(selected_keys)

    return selected_keys

def key_identification_clustering_two(document_vectors, n_cluster_target):

    original_cosine_distance, current_distance, knn_indexes, document_cluster_labels, n_cluster_previous, n_cluster_current = load_function()

    while n_cluster_current > n_cluster_target:

        selected_keys = key_item_selection(current_distance, n_cluster_current)
        
        # update elements lablels

        for i in range(len(document_cluster_labels)):
            min_distance = MAXINT
            for j in selected_keys:
                if current_distance[document_cluster_labels[i]][j] < min_distance:
                    min_distance = current_distance[document_cluster_labels[i]][j]
                    new_cluster_label = j
            document_cluster_labels[i] = new_cluster_label
        
        # update current distance matrix
        # p list in article
        counter = 0
        cluster_members_with_neighbors = {}
        for i in selected_keys:
            temp = list(locate(document_cluster_labels, lambda a: a == i))
            for l in temp:
                for j in range(k):                    
                    temp.append(knn_indexes[l][j])
                temp = list(set(temp))  
            cluster_members_with_neighbors[counter] = temp
            counter += 1

        current_new_distance = np.zeros(shape=(n_cluster_current, n_cluster_current))
        for i in range(n_cluster_current):
            for j in range(n_cluster_current):
                current_distance_sigma = 0

                for l in cluster_members_with_neighbors[i]:
                    for p in cluster_members_with_neighbors[j]:
                        current_distance_sigma += original_cosine_distance[l][p]
                
                current_new_distance[i][j] = current_distance_sigma/(len(cluster_members_with_neighbors[i])*len(cluster_members_with_neighbors[j]))
            
            print("update current distance matrix %", ((i)/(n_cluster_current))*100)

        # update n_cluster_previous & n_cluster_current
        n_cluster_previous = n_cluster_current
        n_cluster_current = math.floor(n_cluster_current/g)
        print("^^^")
        current_distance = current_new_distance

        c = 0
        cluster_label_dict_current = {}
        for i in selected_keys:
            cluster_label_dict_current[i] = c
            c += 1 

        for i in range(len(document_cluster_labels)):
            document_cluster_labels[i] = cluster_label_dict_current[document_cluster_labels[i]]

    selected_keys = key_item_selection(current_distance, n_cluster_target)
    
    # update final elements lablels
    for i in range(len(document_cluster_labels)):
        min_distance = MAXINT
        for j in selected_keys:
            if current_distance[document_cluster_labels[i]][j] < min_distance:
                min_distance = current_distance[document_cluster_labels[i]][j]
                new_cluster_label = j
        document_cluster_labels[i] = new_cluster_label

    c = 0
    cluster_label_dict_current = {}
    for i in selected_keys:
        cluster_label_dict_current[i] = c
        c += 1 
    
    for i in range(len(document_cluster_labels)):
        document_cluster_labels[i] = cluster_label_dict_current[document_cluster_labels[i]]
        
    return document_cluster_labels

k = 3
g = 40
n_cluster_target = 3

model = gensim.models.doc2vec.Doc2Vec.load("step4/doc2vec.model")

labels = pd.read_csv('dataset/train.csv')

key_identification_clusters = key_identification_clustering_two(model.dv, n_cluster_target)

kmeans = KMeans(n_clusters=n_cluster_target)

li = np.zeros(shape=(8695, 65))
x = np.zeros(shape=(8695, 1))
y = np.zeros(shape=(8695, 1))

for i in range(len(model.dv)):
    li[i] = np.array(model.dv[i])
    x[i] = li[i][0]
    y[i] = li[i][1]

kmeans.fit(li)

#kmeans labels
# for i in range(len(kmeans.labels_)):
#     print(kmeans.labels_[i], labels["Topic"].iloc[i])

# Our labels
for i in range(len(key_identification_clusters)):
    print(key_identification_clusters[i], labels["Topic"].iloc[i])

Biology_0 = 0
Biology_1 = 0
Biology_2 = 0

Chemistry_0 = 0
Chemistry_1 = 0
Chemistry_2 = 0

Physics_0 = 0
Physics_1 = 0
Physics_2 = 0

for i in range(len(key_identification_clusters)):
    if key_identification_clusters[i] == 0:

        if labels["Topic"].iloc[i] == "Biology":
            Biology_0 += 1
        if labels["Topic"].iloc[i] == "Chemistry":
            Chemistry_0 += 1
        if labels["Topic"].iloc[i] == "Physics":
            Physics_0 += 1
            
    if key_identification_clusters[i] == 1:

        if labels["Topic"].iloc[i] == "Biology":
            Biology_1 += 1
        if labels["Topic"].iloc[i] == "Chemistry":
            Chemistry_1 += 1
        if labels["Topic"].iloc[i] == "Physics":
            Physics_1 += 1

    if key_identification_clusters[i] == 2:
        
        if labels["Topic"].iloc[i] == "Biology":
            Biology_2 += 1
        if labels["Topic"].iloc[i] == "Chemistry":
            Chemistry_2 += 1
        if labels["Topic"].iloc[i] == "Physics":
            Physics_2 += 1


#kmeans colors
kmeans_colors = kmeans.labels_

#key_identification_cluster colors
key_identification_cluster_colors = np.asarray(key_identification_clusters)

#main colors
main_colors = np.zeros(shape=(8695, 1))

for i in range(len(labels["Topic"])):
    if labels["Topic"].iloc[i] == "Biology":
        main_colors[i] = 0
    if labels["Topic"].iloc[i] == "Chemistry":
        main_colors[i] = 1
    if labels["Topic"].iloc[i] == "Physics":
        main_colors[i] = 2


print("Kmeans Accuracy:", (100*metrics.accuracy_score(main_colors, kmeans_colors)))
print("Key Identification Accuracy:", (100*metrics.accuracy_score(main_colors, key_identification_cluster_colors)))

print("Biology_0", Biology_0)
print("Biology_1", Biology_1)
print("Biology_2", Biology_2)
print("Chemistry_0", Chemistry_0)
print("Chemistry_1", Chemistry_1)
print("Chemistry_2", Chemistry_2)
print("Physics_0", Physics_0)
print("Physics_1", Physics_1)
print("Physics_2", Physics_2)

plt.scatter(x, y, c=kmeans_colors)
plt.show()
plt.scatter(x, y, c=key_identification_cluster_colors)
plt.show()
plt.scatter(x, y, c=main_colors)
plt.show()

# text_array = labels['Comment']
# text_array = [gensim.models.doc2vec.TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(text_array)] 
# documents_tags_dataframe = pd.DataFrame(text_array)
# model = gensim.models.Doc2Vec(dm=1, vector_size=65, hs=1, min_count=2, sample = 12000,window=3, alpha=0.025, min_alpha=0.00025)
# model.build_vocab(text_array)
# model.train(text_array, total_examples=model.corpus_count, epochs=70)
# model.save('step5_2/doc2vec_test.model')



# print(key_identification_clusters)