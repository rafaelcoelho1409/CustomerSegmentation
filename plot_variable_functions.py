import streamlit as st, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def plot_sex(dataframe:pd.DataFrame, options):
    plt.bar(range(len(dataframe['Sex'].value_counts())), dataframe['Sex'].value_counts())
    plt.title(options, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Sexo', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Contagem', fontweight = 'bold', fontsize = 12)
    plt.xticks(range(len(dataframe['Sex'].value_counts())), labels = dataframe['Sex'].value_counts().index.tolist())

def plot_age(dataframe:pd.DataFrame, options):
    age = pd.DataFrame()
    for i in range(2,9):
        age = age.append({'interval': '{}-{}'.format(i*10, (i+1)*10), 'count': dataframe[dataframe['Age'].between(i*10,(i+1)*10)]['Age'].value_counts().sum()}, ignore_index = True)
    plt.bar(range(len(age['interval'].value_counts())), age['count'])
    plt.title(options, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Intervalo de idade', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Contagem', fontweight = 'bold', fontsize = 12)
    plt.xticks(range(len(age['interval'].value_counts())), labels = age['interval'].unique(), fontsize = 12, rotation = 45)

def plot_income(dataframe:pd.DataFrame, slices, options):
    income = pd.DataFrame()
    divs = [((dataframe['Income'].max() - dataframe['Income'].min())/slices)*i for i in range(1,slices + 1)]
    divs_k = divs.copy()
    for i in range(slices):
        divs_k[i] = int(divs_k[i]//1000)
    for j in range(slices - 1):
        income = income.append({'interval': '{}-{}'.format(divs_k[j], divs_k[j+1]), 'count': dataframe[dataframe['Income'].between(divs[j], divs[j+1])].value_counts().sum()}, ignore_index = True)
    plt.bar(range(len(income['interval'].value_counts())), income['count'])
    plt.title(options, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Faixa salarial (US$1000 anuais)', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Contagem', fontweight = 'bold', fontsize = 12)
    plt.xticks(range(len(income['interval'].value_counts())), labels = income['interval'].unique(), fontsize = 12, rotation = 45)

def plot_rating(dataframe:pd.DataFrame, slices, options):
    rating = pd.DataFrame()
    bins = [dataframe['Rating'].min()]
    _range = list(range(dataframe['Rating'].min(), dataframe['Rating'].max()))
    for i in _range:
        if i % slices == 0:
            bins.append(i)
    bins.append(dataframe['Rating'].max())
    for j in range(len(bins) - 1):
        if (bins[j+1] - bins[j]) < slices//2:
            del bins[j]
    for k in range(len(bins) - 1):
        rating = rating.append({'interval': '{}-{}'.format(bins[k], bins[k+1]), 'count': dataframe[dataframe['Rating'].between(bins[k], bins[k+1])].value_counts().sum()}, ignore_index = True)
    plt.bar(range(len(rating['interval'].value_counts())), rating['count'])
    plt.title(options, fontweight = 'bold', fontsize = 15)
    plt.xlabel('Score', fontweight = 'bold', fontsize = 12)
    plt.ylabel('Contagem', fontweight = 'bold', fontsize = 12)
    plt.xticks(range(len(rating['interval'].value_counts())), labels = rating['interval'].unique(), fontsize = 12, rotation = 45)
