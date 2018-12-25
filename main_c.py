# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:40:09 2018

@author: tgill
"""
import numpy as np
from keras.applications.vgg16 import VGG16
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import kmapper as km
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.base import BaseEstimator
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial import distance

def top_dens_nn(kernel, p=0.2, n=5):
    neigh = NearestNeighbors()
    neigh.fit(kernel)
    distances, indices = neigh.kneighbors(kernel, n)
    d = distances[:,n-1]
    idxs = np.argsort(d)
    top20 = idxs[:int(len(kernel)*p)]
    kernel20 = kernel[top20]
    densities = d[top20]
    return kernel20 

def SNE(V):
    def sne(u, v):
        return scipy.spatial.distance.seuclidean(u, v, V)
    return sne

class Single_linkage(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit_predict(self, X):
        d = distance.cdist(X, X, 'seuclidean')
        Z = linkage(d, 'single')
#        plt.figure()
#        dendrogram(Z)    
#        plt.show()
        lens = np.zeros(Z.shape[0] + 1)
        lens[:-1] = Z[:, 2]
        lens[-1] = d.max()
        hst, _ = np.histogram(lens, bins=4)
        z = np.nonzero(hst == 0)[0]
        n_clust = hst[z[0]:len(hst)].sum()
        print(n_clust)
        clustering = cut_tree(2, n_clust)
        return clustering
    
    def fit(self, X):
        if len(X)==1:
            self.labels_=np.asarray([0])
            return self
        d = distance.cdist(X, X, 'seuclidean')
        Z = linkage(d, 'single')
    #        plt.figure()
    #        dendrogram(Z)    
    #        plt.show()
        lens = np.zeros(Z.shape[0] + 1)
        lens[:-1] = Z[:, 2]
        lens[-1] = d.max()
        hst, _ = np.histogram(lens, bins=4)
        z = np.nonzero(hst == 0)[0]
        if len(z)==0:
            n_clust=1
        else:
            n_clust = hst[z[0]:len(hst)].sum()
        #print(n_clust)
        clustering = cut_tree(Z, n_clust)
        self.labels_ = clustering
        return self

def full(n_layer=2, n_file=2):
    #Get weights
    model = VGG16(weights='imagenet', include_top=False)
    weights = model.layers[n_layer].get_weights()[0]
    
    #Reshape weights
    s = weights.shape
    kern = weights.transpose([2,3,0,1]).reshape(s[2]*s[3], s[0]*s[0])
    kern_disp = kern.reshape(len(kern), s[0], s[0])
    
    #Normalize
    kern_means = kern.mean(axis=1)
    kern_std = kern.std(axis=1)
    
    kern_scaled = kern
    kern_scaled = np.asarray([kern_scaled[i]-kern_means[i] for i in range(len(kern_scaled))])
    kern_scaled = np.asarray([kern_scaled[i]/kern_std[i] for i in range(len(kern_scaled))])
    
    #Select top density points
    kern20nn  = top_dens_nn(kern_scaled, n=100, p=0.3)

    name = 'VGG_layer_'+str(n_file)
    
    proj_pca = PCA(n_components=2, whiten=False)
    
    mapper = km.KeplerMapper(verbose=1)    
    lens = mapper.fit_transform(kern20nn, projection=proj_pca, scaler=None, distance_matrix=False)
    
    plt.figure()
    plt.scatter(lens[:,0], lens[:,1], s=5)
    plt.title(name)
    
    V = kern20nn.std(axis=0)
    
#    d = distance.cdist(kern20nn, kern20nn)
#    Z = linkage(d, 'single')
#    plt.figure()
#    dendrogram(Z)    
#    plt.show()
#    lens = np.zeros(Z.shape[0] + 1)
#    lens[:-1] = Z[:, 2]
#    lens[-1] = d.max()
#    hst, bins = np.histogram(lens, bins=64)
#    plt.figure()
#    plt.hist(lens, bins=64)
#    z = np.nonzero(hst == 0)[0]
#    print(hst[z[0]:len(hst)].sum())
#    print(z.shape)
#    print(z[:10])
    
    graph = mapper.map(lens,
                       kern20nn,
                       #clusterer = AgglomerativeClustering(n_clusters=2, linkage='single', affinity='euclidean'),
                       clusterer = Single_linkage(),
                    #   clusterer = DBSCAN(metric=SNE(V)),
                       coverer=km.Cover(nr_cubes=30, overlap_perc=0.66),
                       )
    
    ht=mapper.visualize(graph,
                     path_html=name+".html",
                     title=name
                     )
    
    
    return graph