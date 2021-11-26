import streamlit as st, pandas as pd, matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import silhouette_score
from plot_variable_functions import plot_sex, plot_age, plot_income, plot_rating
from plot_cluster_model_functions import plot_pca, plot_pca_kmeans, plot_pca_agg, plot_pca_dbscan
st.set_page_config(layout = 'wide')

st.title('Segmentação de clientes usando modelos de clusterização')
st.write(
"""
O objetivo deste projeto é analisar os dados de clientes e segmentá-los através de clusterização, 
baseado em sexo, idade, salário (dólares anuais) e score de compras.\n
Para isso, iremos usar alguns algoritmos de clusterização (Machine Learning).\n
Os dados foram extraidos do Kaggle (o link está no arquivo README.md)""")

data = pd.read_csv('amazon.csv')
data['Sex'] = data['Sex'].apply(lambda x: 'Masculino' if x == 'M' else 'Feminino')
#Tratamento dos dados
data2 = data.drop(['Cus_ID'], axis = 1)
data2 = pd.get_dummies(data)
st.write('## Dados que usaremos nas análises e na segmentação')
st.write(data2)

st.write('## Análise exploratória dos dados')
col01, col02 = st.columns(2)
with col01:
    options = st.selectbox('Selecione a variável que você deseja visualizar', ['Idade', 'Sexo', 'Salário', 'Score de compra'])
    st.write('#### Descrição das variáveis')
    st.write(data2.describe())
with col02:
    if options == 'Sexo':
        fig, ax = plt.subplots()
        plot_sex(data, options)
        st.pyplot(fig)
    elif options == 'Idade':
        fig, ax = plt.subplots()
        plot_age(data, options)
        st.pyplot(fig)
    elif options == 'Salário':
        fig, ax = plt.subplots()
        plot_income(data, slices = 10, options = options)
        st.pyplot(fig)
    elif options == 'Score de compra':
        fig, ax = plt.subplots()
        plot_rating(data, slices = 20, options = options)
        st.pyplot(fig)

st.write('## Modelos de clusterização (Machine Learning)')
col11, col12, col13 = st.columns(3)
with col11:
    cluster_model = st.selectbox('Selecione o modelo de clusterização', 
    ['PCA + KMeans', 'PCA + Agglomerative Clustering', 'PCA + DBSCAN', 'Principal Component Analysis (PCA)'])
    scaler_option = st.selectbox('Selecione o escalador (redimensionador dos dados para uma escala menor)', ['MinMaxScaler', 'StandardScaler'])
    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    #########################################
    if cluster_model == 'Principal Component Analysis (PCA)':
        scaler.fit(data2)
        data_scaled = scaler.transform(data2)
        pca = PCA(n_components = 2, random_state = 0)
        pca.fit(data_scaled)
        data_pca = pca.transform(data_scaled)
    elif cluster_model == 'PCA + KMeans':
        number_of_clusters = st.number_input('Selecione o número de clusters (grupos)', min_value = 2, step = 1)
        scaler.fit(data2)
        data_scaled = scaler.transform(data2)
        pca = PCA(n_components = 2, random_state = 0)
        pca.fit(data_scaled)
        data_pca = pca.transform(data_scaled)
        kmeans = KMeans(n_clusters = number_of_clusters, random_state = 0)
        kmeans.fit(data_pca)
        y_pred = kmeans.predict(data_pca)
    elif cluster_model == 'PCA + Agglomerative Clustering':
        number_of_clusters = st.number_input('Selecione o número de clusters (grupos)', min_value = 2, step = 1)
        scaler.fit(data2)
        data_scaled = scaler.transform(data2)
        pca = PCA(n_components = 2, random_state = 0)
        pca.fit(data_scaled)
        data_pca = pca.transform(data_scaled)
        agg = AgglomerativeClustering(n_clusters = number_of_clusters)
        y_pred = agg.fit_predict(data_pca)
    elif cluster_model == 'PCA + DBSCAN':
        eps = st.slider('Raio de distância entre os pontos (eps)', min_value = 0.01, value = float(0.25))
        min_samples = st.number_input('Número mínimo de pontos para cada cluster (min_samples)', min_value = 1, value = 30, step = 1)
        scaler.fit(data2)
        data_scaled = scaler.transform(data2)
        pca = PCA(n_components = 2, random_state = 0)
        pca.fit(data_scaled)
        data_pca = pca.transform(data_scaled)
        dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        y_pred = dbscan.fit_predict(data_pca)
with col12:
    if cluster_model == 'Principal Component Analysis (PCA)':
        fig, ax = plt.subplots()
        plot_pca(cluster_model, data_pca)
        st.pyplot(fig)
    elif cluster_model == 'PCA + KMeans':
        fig, ax = plt.subplots()
        plot_pca_kmeans(kmeans, cluster_model, data_pca, y_pred, number_of_clusters)
        st.pyplot(fig)
    elif cluster_model == 'PCA + Agglomerative Clustering':
        fig, ax = plt.subplots()
        plot_pca_agg(cluster_model, data_pca, y_pred)
        st.pyplot(fig)
    elif cluster_model == 'PCA + DBSCAN':
        fig, ax = plt.subplots()
        plot_pca_dbscan(cluster_model, data_pca, y_pred)
        st.pyplot(fig)
with col13:
    if cluster_model in ['PCA + KMeans', 'PCA + Agglomerative Clustering', 'PCA + DBSCAN']:
        st.write('### Taxa de acerto do modelo')
        st.write('#### Silhouette Score')
        st.write('#### {:.2f}%'.format(silhouette_score(data_pca, y_pred)*100))

if cluster_model not in ['Principal Component Analysis (PCA)', 'PCA + DBSCAN']:
    st.write('## Amostras separadas por cada cluster')
    _list = []
    for i in range(len(np.unique(y_pred))):
        var = 'col2{},'.format(i)
        _list.append(var)
    vars = ''.join(_list)
    st.write('### Analisando as variáveis por cluster')
    globals()['({})'.format(vars)] = st.columns(len(_list))
    for j in range(len(np.unique(y_pred))):
        with globals()['({})'.format(vars)][j]:
            st.write('- #### Cluster {}'.format(j))
            globals()['data_{}'.format(j)] = data[pd.Series(y_pred) == j]
            st.write(globals()['data_{}'.format(j)])
            st.write('\n')
            globals()['variables{}'.format(j)] = st.selectbox('Selecione a variável que você deseja analisar', ['Idade', 'Sexo', 'Salário', 'Score de compra'], key = j)
            if globals()['variables{}'.format(j)] == 'Sexo':
                fig, ax = plt.subplots()
                plot_sex(globals()['data_{}'.format(j)], globals()['variables{}'.format(j)])
                st.pyplot(fig)
            elif globals()['variables{}'.format(j)] == 'Idade':
                fig, ax = plt.subplots()
                plot_age(globals()['data_{}'.format(j)], globals()['variables{}'.format(j)])
                st.pyplot(fig)
            elif globals()['variables{}'.format(j)] == 'Salário':
                fig, ax = plt.subplots()
                plot_income(globals()['data_{}'.format(j)], slices = 10, options = globals()['variables{}'.format(j)])
                st.pyplot(fig)
            elif globals()['variables{}'.format(j)] == 'Score de compra':
                fig, ax = plt.subplots()
                plot_rating(globals()['data_{}'.format(j)], slices = 20, options = globals()['variables{}'.format(j)])
                st.pyplot(fig)
elif cluster_model == 'PCA + DBSCAN':
    st.write('## Amostras separadas por cada cluster')
    _list = []
    for i in range(len(np.unique(y_pred))):
        var = 'col2{},'.format(i)
        _list.append(var)
    vars = ''.join(_list)
    st.write('### Analisando as variáveis por cluster')
    globals()['({})'.format(vars)] = st.columns(len(_list))
    for j in range(len(np.unique(y_pred))):
        with globals()['({})'.format(vars)][j]:
            preds = np.sort(pd.Series(y_pred).unique())
            if preds[j] == -1:
                st.write('- #### Ruído (noise)')
                globals()['data_{}'.format(j)] = data[pd.Series(y_pred) == -1]
                st.write(globals()['data_{}'.format(j)])
                st.write('\n')
                globals()['variables{}'.format(j)] = st.selectbox('Selecione a variável que você deseja analisar', ['Idade', 'Sexo', 'Salário', 'Score de compra'], key = j)
                if globals()['variables{}'.format(j)] == 'Sexo':
                    fig, ax = plt.subplots()
                    plot_sex(globals()['data_{}'.format(j)], options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Idade':
                    fig, ax = plt.subplots()
                    plot_age(globals()['data_{}'.format(j)], options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Salário':
                    fig, ax = plt.subplots()
                    plot_income(globals()['data_{}'.format(j)], slices = 10, options = options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Score de compra':
                    fig, ax = plt.subplots()
                    plot_rating(globals()['data_{}'.format(j)], slices = 20, options = options)
                    st.pyplot(fig)
            else:
                st.write('- #### Cluster {}'.format(j-1))
                globals()['data_{}'.format(j)] = data[pd.Series(y_pred) == j-1]
                st.write(globals()['data_{}'.format(j)])
                st.write('\n')
                globals()['variables{}'.format(j)] = st.selectbox('Selecione a variável que você deseja analisar', ['Idade', 'Sexo', 'Salário', 'Score de compra'], key = j)
                if globals()['variables{}'.format(j)] == 'Sexo':
                    fig, ax = plt.subplots()
                    plot_sex(globals()['data_{}'.format(j)], options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Idade':
                    fig, ax = plt.subplots()
                    plot_age(globals()['data_{}'.format(j)], options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Salário':
                    fig, ax = plt.subplots()
                    plot_income(globals()['data_{}'.format(j)], slices = 10, options = options)
                    st.pyplot(fig)
                elif globals()['variables{}'.format(j)] == 'Score de compra':
                    fig, ax = plt.subplots()
                    plot_rating(globals()['data_{}'.format(j)], slices = 20, options = options)
                    st.pyplot(fig)

