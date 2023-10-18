import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn import metrics


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from scipy import stats

from pandas.api.types import is_string_dtype, is_numeric_dtype


def KMeans_Algorithm(dataset, n):
    clustering_KMeans = KMeans(n_clusters= n,init='k-means++', max_iter=300, random_state=0, algorithm = "elkan")
    clustering_KMeans.fit(dataset)
    
    # create data frame to store centroids
    centroids  = clustering_KMeans.cluster_centers_
    
    # add cluster label for each data point
    label = clustering_KMeans.labels_
    df["label"] = label
    
    # evaluation metrics for clustering - inertia and silhouette score
    inertia = clustering_KMeans.inertia_
    silhouette_score = metrics.silhouette_score(dataset, label)
    
    return inertia, label, centroids, silhouette_score


def data_scaler(scaler, var):
    scaled_var = "scaled_" + var
    model = scaler.fit(df[var].values.reshape(-1,1))
    df[scaled_var] = model.transform(df[var].values.reshape(-1, 1))
    
    plt.figure(figsize = (5,5))
    plt.title(scaled_var)
    df[scaled_var].plot(kind = 'hist')
    
    plt.figure(figsize = (5,5))
    plt.title(var)
    df[var].plot(kind = 'hist')

# Título do aplicativo
st.title('Aplicativo de Análise de Dadosssss')


# Carregar o arquivo CSV
file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])
if file is not None:
    df = pd.read_csv(file, sep=';')

    # Mostrar o DataFrame original
    st.write('DataFrame Original:')
    st.write(df)

    # Mostrar o tipo de cada coluna
    st.write('Tipos de Colunas:')
    st.write(df.dtypes)

    # Inicializar um DataFrame vazio para armazenar os resultados
    converted_df = pd.DataFrame()

    selected_columns = st.multiselect('Selecione as Colunas', df.columns)

    if not selected_columns:
        st.warning('Por favor Selecione as colunas')
    else:
        # Loop através das colunas e permitir a conversão
        for col in selected_columns:
            conversion_type = st.selectbox(f'Tipo desejado para {col}:', ['int', 'float', 'string'])

            if conversion_type == 'int':
                converted_df[col] = df[col].str.replace(',', '.').astype(int)
            elif conversion_type == 'float':
                converted_df[col] = df[col].str.replace(',', '.').astype(float)
            elif conversion_type == 'string':
                converted_df[col] = df[col].astype(str)

    # Mostrar o DataFrame transformado
    st.write('DataFrame Transformado:')
    st.write(converted_df)
    df = converted_df

    st.subheader('Normalizar os Dados')

    # populate list of numerical and categorical variables
    num_list = []
    cat_list = []

    for column in df:
        if is_numeric_dtype(df[column]):
            num_list.append(column)
        elif is_string_dtype(df[column]):
            cat_list.append(column)
        

    st.write("Variáveis Numéricas:", num_list)
    st.write("variáveis Categóricas:", cat_list)

    if st.checkbox('Retirar Dados Nulos'):
        df = df.dropna()
        df = df.drop(df[(df == 0).any(axis=1)].index)


    if st.checkbox('Normalizar Dados Númericos'):

        scaler = MinMaxScaler()

        for var in num_list:
            scaled_var = "scaled_" + var
            model = scaler.fit(df[var].values.reshape(-1,1))
            df[scaled_var] = model.transform(df[var].values.reshape(-1, 1))

    if st.checkbox('Retirar Outliers'):

        if st.checkbox('Deseja remover o volume dos Outliers?'): 
            nova_lista = [item for item in num_list if 'Volume (t)' not in item]

            st.write('Lista de Variáveis no filtro de Outliers')
            st.write(nova_lista)
            st.write('Outliers') 
            df[~(np.abs(stats.zscore(df[nova_lista])) < 3).all(axis=1)]
            st.write('Tabela Resultante') 
            df = df[(np.abs(stats.zscore(df[nova_lista])) < 3).all(axis=1)]
        else:
            st.write('Lista de Variáveis no filtro de Outliers')
            st.write(num_list)
            st.write('Outliers') 
            df[~(np.abs(stats.zscore(df[num_list])) < 3).all(axis=1)]
            st.write('Tabela Resultante') 
            df = df[(np.abs(stats.zscore(df[num_list])) < 3).all(axis=1)]

    st.write(df)  

    st.subheader('Inicializar o Agrupamento:')

    col_x = st.selectbox('Coluna 1',df.columns)
    col_y = st.selectbox('Coluna 2',df.columns)
    col_z = st.selectbox('Coluna 3',df.columns)

    X4 = df[[col_x, col_y, col_z]].values


    if st.button('Testar K-Means'):
        
        X4_inertia_values = []
        X4_silhouette_scores = []
        fig = plt.figure(figsize=(24,24))
        for i in range (2,11):
            X4_inertia, X4_label, X4_centroids, X4_silhouette = KMeans_Algorithm(X4, i)
            X4_inertia_values.append(X4_inertia)
            X4_silhouette_scores.append(X4_silhouette)
            centroids_df = pd.DataFrame(X4_centroids, columns =['X', 'Y', 'Z'])
            ax = fig.add_subplot(330 + i - 1, projection='3d')
            ax.scatter(df[col_x],df[col_y],df[col_z], s = 30, c = df["label"], cmap = "RdBu")
            ax.scatter(centroids_df['X'], centroids_df['Y'], centroids_df['Z'], s = 180, marker= ",", color = "r")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_zlabel(col_z)
            
        st.pyplot(fig)
        

        # Crie uma figura e plote o gráfico
        fig1, ax = plt.subplots(figsize=(15, 6))
        ax.plot(np.arange(2, 11), X4_inertia_values, '-')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Inertia Values")

        # Use st.pyplot para exibir o gráfico no Streamlit
        st.pyplot(fig1)

        # Crie uma figura e plote o gráfico
        fig2, ax = plt.subplots(figsize=(15, 6))
        ax.plot(np.arange(2, 11), X4_silhouette_scores, '-')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Silhouette Score")

        # Use st.pyplot para exibir o gráfico no Streamlit
        st.pyplot(fig2)


        # if st.button('Gerar Agrupamento'):
            
    st.subheader('Simulação de Grupos:')

    valor_selecionado = st.slider("Escolha o número de Grupos:", min_value=2, max_value=11, step=1)

    X4_inertia, X4_label, X4_centroids, X4_silhouette = KMeans_Algorithm(X4, valor_selecionado)
    df['Groups']= X4_label

    # Crie o gráfico Plotly
    fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z,
                color='Groups',  hover_name="Ult.Customer - Live", size_max=16, opacity=0.5, color_continuous_scale='Portland')

    # Defina o fundo como branco
    fig.update_layout(scene=dict(aspectmode="cube", xaxis_title=col_x, yaxis_title=col_y, zaxis_title=col_z,
                                bgcolor='white'))  

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Use st.pyplot para exibir o gráfico no Streamlit
    st.write(fig)

    # df.drop('label', inplace=True)
    st.download_button(
        label="Baixar Tabela",
        data = df.to_csv(index=False, sep=';'),
        file_name='kmeans_results.csv',
        key='download_button'
    )
    
    


