# Segmentação de clientes usando algoritmos de clusterização (Machine Learning)

## Objetivo
- O objetivo deste projeto é analisar os dados de clientes e segmentá-los através de modelos de clusterização, 
baseado em sexo, idade, salário (dólares anuais) e score de compras.

## Link deste projeto:
- https://share.streamlit.io/rafaelcoelho1409/customersegmentation/customer_segmentation.py

## Recursos utilizados
- Visual Studio Code
- python3.8
- virtualenv
- pip3: gerenciador de pacotes python3.x

## Pacotes do Python
- streamlit
- pandas
- plotly
- numpy
- sklearn (scikit-learn) 

## Algoritmos de clusterização usados
- Principal Component Analysis (PCA)
- PCA + KMeans
- PCA + Agglomerative Clustering

## Dataset usado para desenvolver o projeto
- https://www.kaggle.com/harshsoni7254/amazon-customer-segmentation

## Para executar esse arquivo localmente em sua máquina
- baixe esse repositório em sua máquina:
> git clone https://github.com/rafaelcoelho1409/CustomerSegmentation.git
- instale os pacotes necessários que estão no arquivo requirements.txt:
> pip3 install -r requirements.txt
- escolha seu interpretador python (python3, python3.x)  
- execute os seguintes comandos (para Linux):
> cd CustomerSegmentation  
> streamlit run customer_segmentation.py  
- Com esses comandos, a página será aberta automaticamente. Caso não abra, vá até seu navegador e digite:
> http://localhost:8501

## Comentários sobre o projeto
- Na parte onde você aplica algum algoritmo de clusterização, conforme você aumenta o número de clusters, automaticamente a página mostra os dados associados a cada cluster e os gráficos associados. 
- Para visualização da análise dos dados e das predições usando os algoritmos de Machine Learning, usei o pacote Streamlit, que permite que você crie uma página web dinâmica localmente onde você pode inserir qualquer coisa que use Python. Esta ferramenta não só facilita muito as análises dos dados como também permite com que o usuário possa interagir com a página. 
- Essa interatividade permite que usar Streamlit, em vários casos, seja melhor que usar Jupyter Notebook, que é um arquivo estático e dependendo da sua análise, pode acabar ficando muito longo para uma apresentação final.
- O projeto está passível de melhorias, tanto na visualização de dados como no código fonte dele. Tive alguns pequenos problemas com o jeito que escrevi o código, porém quando lembrei de usar Programação Orientada a Objeto usando classes (como já fiz com o GoogleShoppingBot), eu já tinha escrito praticamente tudo o que precisava pra rodar o código. Fica de lição para próximos projetos.

## Melhorias a serem feitas
- Inserir esse projeto dentro de um contâiner Docker

## Screenshots da página construída
<img src="images/001.png" width="800" height="400"/>
<img src="images/002.png" width="800" height="400"/>
<img src="images/003.png" width="800" height="400"/>
<img src="images/004.png" width="800" height="400"/>
<img src="images/005.png" width="800" height="400"/>
<img src="images/006.png" width="800" height="400"/>
<img src="images/007.png" width="800" height="400"/>