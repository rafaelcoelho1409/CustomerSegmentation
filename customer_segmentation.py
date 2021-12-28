import streamlit as st, pandas as pd, plotly.express as px, numpy as np
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.cluster import silhouette_score

st.set_page_config(layout = 'wide')

st.title('Segmentação de clientes usando modelos de clusterização')
st.write(
"""
O objetivo deste projeto é analisar os dados de clientes e segmentá-los através de clusterização, 
baseado em sexo, idade, salário (dólares anuais) e score de compras.\n
Para isso, iremos usar alguns algoritmos de clusterização (Machine Learning).\n
Os dados foram extraidos do Kaggle (o link está no arquivo README.md)""")

#Leitura dos dados
data = pd.read_csv('amazon.csv')
data['Sexo'] = data['Sexo'].apply(lambda x: 'Masculino' if x == 'M' else 'Feminino')
#Tratamento dos dados
data2 = data.drop(['ID de cliente'], axis = 1)
data2 = pd.get_dummies(data)

#Exibição dos dados
with st.expander('Dados de análise para segmentação', expanded = True):
    st.header('Dados de análise para segmentação')
    st.dataframe(data)

with st.expander('Estatística dos dados', expanded = True):
    st.header('Estatística dos dados')
    col01, col02 = st.columns(2)
    with col01:
        options = st.selectbox('Selecione a variável que você deseja visualizar', ['Idade', 'Sexo', 'Salário', 'Score de compra'])
        st.subheader('Descrição das variáveis')
        st.dataframe(data2.describe())
    with col02:
        if options == 'Sexo':
            sexo_df = data[['ID de cliente', 'Sexo']].groupby(['Sexo']).size().reset_index().rename(columns = {0: 'quantidade'})
            fig = px.bar(sexo_df, x = 'Sexo', y = 'quantidade', color = 'Sexo', title = '{}'.format(options))
        elif options == 'Idade':
            idade_df = pd.DataFrame()
            for i in range(2,9):
                idade_df = idade_df.append({'Intervalo': '{}-{}'.format(i*10, (i+1)*10), 'quantidade': data[data['Idade'].between(i*10,(i+1)*10)]['Idade'].value_counts().sum()}, ignore_index = True)
            fig = px.bar(idade_df, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{}'.format(options))
        elif options == 'Salário':
            salario_df = pd.DataFrame()
            #fracionando intervalos de salário em dez faixas
            divs = [((data['Salário'].max() - data['Salário'].min())/10)*i for i in range(1,10 + 1)]
            divs_k = divs.copy()
            for i in range(10):
                divs_k[i] = int(divs_k[i]//1000)
            for j in range(10 - 1):
                salario_df = salario_df.append({'Intervalo': '{}-{}'.format(divs_k[j], divs_k[j+1]), 'quantidade': data[data['Salário'].between(divs[j], divs[j+1])].value_counts().sum()}, ignore_index = True)
            fig = px.bar(salario_df, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{} (1000 US$ / ano)'.format(options))
        elif options == 'Score de compra':
            score_df = pd.DataFrame()
            bins = [data['Score'].min()]
            _range = list(range(data['Score'].min(), data['Score'].max()))
            #dividindo intervalos de em faixas de 10 unidades
            for i in _range:
                if i % 10 == 0:
                    bins.append(i)
            bins.append(data['Score'].max())
            for j in range(len(bins) - 1):
                if (bins[j+1] - bins[j]) < 10//2:
                    del bins[j]
            for k in range(len(bins) - 1):
                score_df = score_df.append({'Intervalo': '{}-{}'.format(bins[k], bins[k+1]), 'quantidade': data[data['Score'].between(bins[k], bins[k+1])].value_counts().sum()}, ignore_index = True)
            fig = px.bar(score_df, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{}'.format(options))
        #plotando cada gráfico
        st.plotly_chart(fig)

with st.expander('Modelos de clusterização (Machine Learning)', expanded = True):
    st.header('Modelos de clusterização (Machine Learning)')
    col11, col12, col13 = st.columns(3)
    with col11:
        cluster_model = st.selectbox('Selecione o modelo de clusterização', 
        ['PCA + KMeans', 'PCA + Agglomerative Clustering', 'Principal Component Analysis (PCA)'])
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
    with col12:
        if cluster_model == 'Principal Component Analysis (PCA)':
            fig = px.scatter(data_pca, x = 0, y = 1, title = cluster_model)
        elif cluster_model == 'PCA + KMeans':
            fig = px.scatter(data_pca, x = 0, y = 1, color = y_pred, title = cluster_model)
            for i in range(kmeans.cluster_centers_.shape[0]):
                fig.add_scatter(x = [kmeans.cluster_centers_[i][0]], y = [kmeans.cluster_centers_[i][1]], marker = {'symbol': 'x', 'size': 15, 'color': 'black'}, name = 'center {}'.format(i))
            fig.update_traces(marker_coloraxis = None)
        elif cluster_model == 'PCA + Agglomerative Clustering':
            fig = px.scatter(data_pca, x = 0, y = 1, color = y_pred, title = cluster_model)
            fig.update_traces(marker_coloraxis = None)
        #plotando os gráficos
        fig.update_layout(height = 400, width = 550)
        st.plotly_chart(fig, height = 400, width = 550)
    with col13:
        if cluster_model in ['PCA + KMeans', 'PCA + Agglomerative Clustering']:
            st.metric('Taxa de acerto (Silhouette Score)', '{:.2f}%'.format(silhouette_score(data_pca, y_pred)*100))

with st.expander('Amostras separadas por cada cluster', expanded = True):
    if cluster_model not in ['Principal Component Analysis (PCA)']:
        st.header('Amostras separadas por cada cluster')
        st.subheader('Analisando as variáveis por cluster')
        #np.unique(y_pred): número total de clusters
        for i in range(len(np.unique(y_pred))):
            st.write('- #### Cluster {}'.format(i))
            globals()['data_{}'.format(i)] = data[pd.Series(y_pred) == i]
            globals()['(col20_{}, col21_{})'.format(i, i)] = st.columns(2)
            with globals()['(col20_{}, col21_{})'.format(i, i)][0]:
                st.dataframe(globals()['data_{}'.format(i)])
            with globals()['(col20_{}, col21_{})'.format(i, i)][1]:
                globals()['variables_{}'.format(i)] = st.selectbox('Selecione a variável que você deseja analisar', ['Idade', 'Sexo', 'Salário', 'Score de compra'], key = i)
                if globals()['variables_{}'.format(i)] == 'Sexo':
                    sexo_df2 = globals()['data_{}'.format(i)][['ID de cliente', 'Sexo']].groupby(['Sexo']).size().reset_index().rename(columns = {0: 'quantidade'})
                    fig = px.bar(sexo_df2, x = 'Sexo', y = 'quantidade', color = 'Sexo', title = '{}'.format(globals()['variables_{}'.format(i)]))
                elif globals()['variables_{}'.format(i)] == 'Idade':
                    idade_df2 = pd.DataFrame()
                    for j in range(2,9):
                        idade_df2 = idade_df2.append({'Intervalo': '{}-{}'.format(j*10, (j+1)*10), 'quantidade': globals()['data_{}'.format(i)][globals()['data_{}'.format(i)]['Idade'].between(j*10,(j+1)*10)]['Idade'].value_counts().sum()}, ignore_index = True)
                    fig = px.bar(idade_df2, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{}'.format(globals()['variables_{}'.format(i)]))
                elif globals()['variables_{}'.format(i)] == 'Salário':
                    salario_df2 = pd.DataFrame()
                    divs = [((globals()['data_{}'.format(i)]['Salário'].max() - globals()['data_{}'.format(i)]['Salário'].min())/10)*j for j in range(1,10 + 1)]
                    divs_k = divs.copy()
                    for j in range(10):
                        divs_k[j] = int(divs_k[j]//1000)
                    for k in range(10 - 1):
                        salario_df2 = salario_df2.append({'Intervalo': '{}-{}'.format(divs_k[k], divs_k[k+1]), 'quantidade': globals()['data_{}'.format(i)][globals()['data_{}'.format(i)]['Salário'].between(divs[k], divs[k+1])].value_counts().sum()}, ignore_index = True)
                    fig = px.bar(salario_df2, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{} (1000 US$ / ano)'.format(globals()['variables_{}'.format(i)]))
                elif globals()['variables_{}'.format(i)] == 'Score de compra':
                    score_df2 = pd.DataFrame()
                    try:
                        bins = [globals()['data_{}'.format(i)]['Score'].min()]
                        _range = list(range(globals()['data_{}'.format(i)]['Score'].min(), globals()['data_{}'.format(i)]['Score'].max()))
                        for j in _range:
                            if j % 20 == 0:
                                bins.append(j)
                        bins.append(globals()['data_{}'.format(i)]['Score'].max())
                        for l in range(len(bins) - 1):
                            if (bins[l+1] - bins[l]) < 20//2:
                                del bins[l]
                    except:
                        try:
                            bins = [globals()['data_{}'.format(i)]['Score'].min()]
                            _range = list(range(globals()['data_{}'.format(i)]['Score'].min(), globals()['data_{}'.format(i)]['Score'].max()))
                            for j in _range:
                                if j % 15 == 0:
                                    bins.append(j)
                            bins.append(globals()['data_{}'.format(i)]['Score'].max())
                            for l in range(len(bins) - 1):
                                if (bins[l+1] - bins[l]) < 15//2:
                                    del bins[l]
                        except:
                            try:
                                bins = [globals()['data_{}'.format(i)]['Score'].min()]
                                _range = list(range(globals()['data_{}'.format(i)]['Score'].min(), globals()['data_{}'.format(i)]['Score'].max()))
                                for j in _range:
                                    if j % 10 == 0:
                                        bins.append(j)
                                bins.append(globals()['data_{}'.format(i)]['Score'].max())
                                for l in range(len(bins) - 1):
                                    if (bins[l+1] - bins[l]) < 10//2:
                                        del bins[l]
                            except:
                                try:
                                    bins = [globals()['data_{}'.format(i)]['Score'].min()]
                                    _range = list(range(globals()['data_{}'.format(i)]['Score'].min(), globals()['data_{}'.format(i)]['Score'].max()))
                                    for j in _range:
                                        if j % 5 == 0:
                                            bins.append(j)
                                    bins.append(globals()['data_{}'.format(i)]['Score'].max())
                                    for l in range(len(bins) - 1):
                                        if (bins[l+1] - bins[l]) < 5//2:
                                            del bins[l]
                                except:
                                    pass
                    for k in range(len(bins) - 1):
                        score_df2 = score_df2.append({'Intervalo': '{}-{}'.format(bins[k], bins[k+1]), 'quantidade': globals()['data_{}'.format(i)][globals()['data_{}'.format(i)]['Score'].between(bins[k], bins[k+1])].value_counts().sum()}, ignore_index = True)
                    fig = px.bar(score_df2, x = 'Intervalo', y = 'quantidade', color = 'Intervalo', title = '{}'.format(globals()['variables_{}'.format(i)]))
                st.plotly_chart(fig)
    else:
        st.write('Não há estatística para ser mostrada.')

with st.expander('Código Python deste dashboard'):
    st.header('Código Python deste dashboard')
    file = open('customer_segmentation.py', 'r').read()
    st.code(file, language = 'python')
