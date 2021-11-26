import streamlit as st, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def plot_pca(cluster_model, data_pca):
    plt.scatter(data_pca[:,0], data_pca[:,1])
    plt.title(cluster_model, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Principal Component 1', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Principal Component 2', fontweight = 'bold', fontsize = 12)

def plot_pca_kmeans(kmeans, cluster_model, data_pca, y_pred, number_of_clusters):
    plt.scatter(data_pca[:,0], data_pca[:,1], c = y_pred)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker = '^', c = list(range(number_of_clusters)))
    plt.title(cluster_model, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Feature 0', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Feature 1', fontweight = 'bold', fontsize = 12)

def plot_pca_agg(cluster_model, data_pca, y_pred):
    plt.scatter(data_pca[:,0], data_pca[:,1], c = y_pred)
    plt.title(cluster_model, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Feature 0', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Feature 1', fontweight = 'bold', fontsize = 12)

def plot_pca_dbscan(cluster_model, data_pca, y_pred):
    plt.scatter(data_pca[:,0], data_pca[:,1], c = y_pred)
    plt.title(cluster_model, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Feature 0', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Feature 1', fontweight = 'bold', fontsize = 12)