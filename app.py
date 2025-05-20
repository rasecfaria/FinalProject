#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Recomendação de Filmes - Aplicação Streamlit
======================================================
Interface web para o sistema de recomendação de filmes usando Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuração da página
st.set_page_config(
    page_title="Sistema de Recomendação de Filmes 🎬",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Sistema de Recomendação de Filmes 🎬")
st.markdown("""
Esta aplicação usa técnicas de machine learning para recomendar filmes baseados em:
1. **Filtragem colaborativa** - baseada nas avaliações de outros usuários
2. **Recomendação baseada em conteúdo** - usando gêneros e características dos filmes
""")

# Funções do sistema de recomendação
@st.cache_data
def carregar_dados(diretorio_dados='ml-latest-small'):
    """
    Carrega os datasets necessários para o sistema de recomendação
    """
    # Caminhos para os arquivos
    path_movies = os.path.join(diretorio_dados, 'movies.csv')
    path_ratings = os.path.join(diretorio_dados, 'ratings.csv')
    path_tags = os.path.join(diretorio_dados, 'tags.csv')
    path_dados = os.path.join(diretorio_dados, 'dados.csv')
    
    # Carregar datasets
    movies = pd.read_csv(path_movies)
    ratings = pd.read_csv(path_ratings)
    
    # Remover timestamp dos ratings
    ratings = ratings[['userId', 'movieId', 'rating']]
    
    # Carregar dados adicionais para recomendação baseada em conteúdo
    try:
        tags = pd.read_csv(path_tags)
        dados = pd.read_csv(path_dados)
        return movies, ratings, tags, dados, True
    except Exception as e:
        st.warning(f"Aviso: Não foi possível carregar dados adicionais: {e}")
        return movies, ratings, None, None, False

@st.cache_data
def preparar_dados_colaborativos(movies, ratings):
    """
    Prepara os dados para filtragem colaborativa
    """
    # Mesclar datasets de filmes e avaliações
    df = movies.merge(ratings, on='movieId')
    
    # Criar tabela pivot: filmes x usuários com valores de avaliações
    movies_table = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    return movies_table

@st.cache_data
def filtragem_colaborativa(movies_table, filme_referencia):
    """
    Implementa o método de filtragem colaborativa usando similaridade de cosseno
    """
    # Calcular similaridade de cosseno entre todos os filmes
    rec = cosine_similarity(movies_table)
    rec_df = pd.DataFrame(rec, columns=movies_table.index, index=movies_table.index)
    
    # Filtrar e ordenar filmes similares ao filme de referência
    if filme_referencia in rec_df.columns:
        cossine_df = pd.DataFrame(rec_df[filme_referencia].sort_values(ascending=False))
        cossine_df.columns = ['recomendações']
        return cossine_df
    else:
        return None

@st.cache_data
def preparar_dados_conteudo(movies, tags, dados):
    """
    Prepara os dados para recomendação baseada em conteúdo
    """
    # Converter movieId para string para garantir o merge correto
    movies['movieId'] = movies['movieId'].apply(lambda x: str(x))
    
    # Converter movieId em tags também para evitar erros de tipo
    tags_copy = tags.copy()
    tags_copy['movieId'] = tags_copy['movieId'].apply(lambda x: str(x))
    
    # Mesclar datasets para criar um único DataFrame enriquecido
    df2 = movies.merge(dados, left_on='title', right_on='Name', how='left')
    df2 = df2.merge(tags_copy, left_on='movieId', right_on='movieId', how='left')
    
    # Usar gêneros como base para informações de conteúdo
    df2['Infos'] = df2['genres']  # + str(df2['Discription']) + df2['tag']
    
    return df2

@st.cache_data
def recomendacao_baseada_em_conteudo(df2, filme_referencia):
    """
    Implementa o método de recomendação baseada em conteúdo usando TF-IDF e similaridade de cosseno
    """
    # Criar matriz TF-IDF
    vec = TfidfVectorizer()
    tfidf = vec.fit_transform(df2['Infos'].apply(lambda x: np.str_(x)))
    
    # Calcular similaridade de cosseno
    sim = cosine_similarity(tfidf)
    sim_df2 = pd.DataFrame(sim, columns=df2['title'], index=df2['title'])
    
    # Filtrar e ordenar filmes similares ao filme de referência
    if filme_referencia in sim_df2.columns:
        final_df = pd.DataFrame(sim_df2[filme_referencia].sort_values(ascending=False))
        final_df.columns = ['recomendações']
        return final_df
    else:
        return None

def mostrar_recomendacoes(recomendacoes, n=10):
    """
    Exibe as recomendações de filmes
    """
    if recomendacoes is not None and not recomendacoes.empty:
        return recomendacoes.head(n)
    else:
        return None

def listar_filmes_populares(movies, ratings, n=20):
    """
    Lista os filmes mais populares baseado no número de avaliações
    """
    # Contar o número de avaliações por filme
    filmes_populares = ratings['movieId'].value_counts().reset_index()
    filmes_populares.columns = ['movieId', 'num_avaliacoes']
    
    # Garantir que movieId tenha o mesmo tipo nos dois dataframes
    filmes_populares['movieId'] = filmes_populares['movieId'].astype(str)
    movies_copy = movies.copy()
    movies_copy['movieId'] = movies_copy['movieId'].astype(str)
    
    # Unir com informações dos filmes
    filmes_populares = filmes_populares.merge(movies_copy[['movieId', 'title']], on='movieId')
    
    # Ordenar por número de avaliações
    filmes_populares = filmes_populares.sort_values('num_avaliacoes', ascending=False)
    
    return filmes_populares.head(n)

def pesquisar_filme(movies, termo_pesquisa):
    """
    Pesquisa filmes por termo no título
    """
    try:
        # Pesquisar de forma case-insensitive e com tratamento para valores NA
        resultados = movies[movies['title'].str.contains(termo_pesquisa, case=False, na=False)]
        
        if resultados.empty:
            return None
        
        return resultados
    except Exception as e:
        st.error(f"Erro ao pesquisar filmes: {e}")
        return None

# Carregar os dados
movies, ratings, tags, dados, conteudo_disponivel = carregar_dados()
movies_table = preparar_dados_colaborativos(movies, ratings)

if conteudo_disponivel:
    df2 = preparar_dados_conteudo(movies, tags, dados)

# Sidebar para seleção de opções
st.sidebar.header("Opções")
opcao = st.sidebar.selectbox(
    "Escolha uma opção",
    ["Ver filmes populares", "Pesquisar filme", "Selecionar filme da lista"]
)

# Implementação das opções
if opcao == "Ver filmes populares":
    st.header("Filmes Populares")
    filmes_populares = listar_filmes_populares(movies, ratings)
    
    # Exibir filmes populares com um seletor para escolher um filme
    st.write("Escolha um filme da lista de filmes populares para receber recomendações:")
    filme_selecionado = st.selectbox("Selecione um filme:", filmes_populares['title'])
    
    if st.button("Recomendar filmes similares"):
        st.header(f"Recomendações para: {filme_selecionado}")
        
        # Método de filtragem colaborativa
        recomendacoes_colaborativas = filtragem_colaborativa(movies_table, filme_selecionado)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseado em Avaliações de Usuários")
            if recomendacoes_colaborativas is not None:
                st.dataframe(mostrar_recomendacoes(recomendacoes_colaborativas, 10))
            else:
                st.info("Não foi possível gerar recomendações baseadas em avaliações.")
        
        # Método baseado em conteúdo
        if conteudo_disponivel:
            recomendacoes_conteudo = recomendacao_baseada_em_conteudo(df2, filme_selecionado)
            
            with col2:
                st.subheader("Baseado em Gêneros e Características")
                if recomendacoes_conteudo is not None:
                    st.dataframe(mostrar_recomendacoes(recomendacoes_conteudo, 10))
                else:
                    st.info("Não foi possível gerar recomendações baseadas em conteúdo.")

elif opcao == "Pesquisar filme":
    st.header("Pesquisar Filmes")
    termo_pesquisa = st.text_input("Digite parte do título do filme:")
    
    if termo_pesquisa:
        resultados = pesquisar_filme(movies, termo_pesquisa)
        
        if resultados is not None and not resultados.empty:
            st.success(f"Encontrados {len(resultados)} filmes com '{termo_pesquisa}'")
            
            filme_selecionado = st.selectbox("Selecione um filme para recomendações:", resultados['title'])
            
            if st.button("Recomendar filmes similares"):
                st.header(f"Recomendações para: {filme_selecionado}")
                
                # Método de filtragem colaborativa
                recomendacoes_colaborativas = filtragem_colaborativa(movies_table, filme_selecionado)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Baseado em Avaliações de Usuários")
                    if recomendacoes_colaborativas is not None:
                        st.dataframe(mostrar_recomendacoes(recomendacoes_colaborativas, 10))
                    else:
                        st.info("Não foi possível gerar recomendações baseadas em avaliações.")
                
                # Método baseado em conteúdo
                if conteudo_disponivel:
                    recomendacoes_conteudo = recomendacao_baseada_em_conteudo(df2, filme_selecionado)
                    
                    with col2:
                        st.subheader("Baseado em Gêneros e Características")
                        if recomendacoes_conteudo is not None:
                            st.dataframe(mostrar_recomendacoes(recomendacoes_conteudo, 10))
                        else:
                            st.info("Não foi possível gerar recomendações baseadas em conteúdo.")
        else:
            st.warning(f"Nenhum filme encontrado contendo '{termo_pesquisa}'")

else:  # Selecionar filme da lista
    st.header("Selecionar Filme da Lista")
    
    # Criar uma lista de todos os filmes ordenados por título
    filmes_ordenados = movies.sort_values(by='title')
    filme_selecionado = st.selectbox("Selecione um filme:", filmes_ordenados['title'])
    
    if st.button("Recomendar filmes similares"):
        st.header(f"Recomendações para: {filme_selecionado}")
        
        # Método de filtragem colaborativa
        recomendacoes_colaborativas = filtragem_colaborativa(movies_table, filme_selecionado)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Baseado em Avaliações de Usuários")
            if recomendacoes_colaborativas is not None:
                st.dataframe(mostrar_recomendacoes(recomendacoes_colaborativas, 10))
            else:
                st.info("Não foi possível gerar recomendações baseadas em avaliações.")
        
        # Método baseado em conteúdo
        if conteudo_disponivel:
            recomendacoes_conteudo = recomendacao_baseada_em_conteudo(df2, filme_selecionado)
            
            with col2:
                st.subheader("Baseado em Gêneros e Características")
                if recomendacoes_conteudo is not None:
                    st.dataframe(mostrar_recomendacoes(recomendacoes_conteudo, 10))
                else:
                    st.info("Não foi possível gerar recomendações baseadas em conteúdo.")

# Mostrar informações adicionais no rodapé
st.markdown("---")
st.markdown("""
### Sobre o Sistema de Recomendação

Este sistema utiliza dados do [MovieLens](https://grouplens.org/datasets/movielens/) e implementa dois métodos de recomendação:

1. **Filtragem Colaborativa**: Baseada nas avaliações dos usuários, usando similaridade de cosseno para encontrar filmes similares.
   - Funciona bem quando temos muitas avaliações, mas pode sofrer com o "cold start" para novos filmes ou usuários.

2. **Recomendação Baseada em Conteúdo**: Utilizando gêneros e características dos filmes.
   - Funciona bem mesmo para novos filmes, desde que tenhamos informações sobre seu conteúdo.

Ambos os métodos se complementam para oferecer recomendações mais precisas e diversificadas.
""")