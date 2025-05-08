#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Recomendação de Filmes
================================
Este script implementa dois métodos de recomendação:
1. Filtragem colaborativa (baseada nas avaliações de usuários)
2. Recomendação baseada em conteúdo (gêneros e tags de filmes)
"""

# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def carregar_dados(diretorio_dados='ml-latest-small'):
    """
    Carrega os datasets necessários para o sistema de recomendação
    
    Args:
        diretorio_dados: Pasta onde os arquivos CSV estão armazenados
        
    Returns:
        Tupla contendo os dataframes carregados
    """
    print("Carregando dados...")
    
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
    
    # Informações sobre os dados carregados
    print(f"Dimensões do dataset de filmes: {movies.shape}")
    print(f"Dimensões do dataset de avaliações: {ratings.shape}")
    
    # Carregar dados adicionais para recomendação baseada em conteúdo
    try:
        tags = pd.read_csv(path_tags)
        dados = pd.read_csv(path_dados)
        print(f"Dimensões do dataset de tags: {tags.shape}")
        print(f"Dados adicionais carregados com sucesso")
        return movies, ratings, tags, dados
    except Exception as e:
        print(f"Aviso: Não foi possível carregar dados adicionais: {e}")
        return movies, ratings, None, None

def preparar_dados_colaborativos(movies, ratings):
    """
    Prepara os dados para filtragem colaborativa
    
    Args:
        movies: DataFrame contendo informações dos filmes
        ratings: DataFrame contendo as avaliações
        
    Returns:
        Tabela pivot de filmes x usuários com as avaliações
    """
    print("Preparando dados para filtragem colaborativa...")
    
    # Mesclar datasets de filmes e avaliações
    df = movies.merge(ratings, on='movieId')
    
    # Criar tabela pivot: filmes x usuários com valores de avaliações
    movies_table = df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
    
    print(f"Tabela pivot criada com dimensões: {movies_table.shape}")
    return movies_table

def filtragem_colaborativa(movies_table, filme_referencia='Forrest Gump'):
    """
    Implementa o método de filtragem colaborativa usando similaridade de cosseno
    
    Args:
        movies_table: Tabela pivot de filmes x usuários
        filme_referencia: Nome do filme para recomendar similares
        
    Returns:
        DataFrame com filmes similares ordenados
    """
    print(f"Calculando filmes similares a '{filme_referencia}' usando filtragem colaborativa...")
    
    # Calcular similaridade de cosseno entre todos os filmes
    from sklearn.metrics.pairwise import cosine_similarity
    rec = cosine_similarity(movies_table)
    rec_df = pd.DataFrame(rec, columns=movies_table.index, index=movies_table.index)
    
    # Filtrar e ordenar filmes similares ao filme de referência
    if filme_referencia in rec_df.columns:
        cossine_df = pd.DataFrame(rec_df[filme_referencia].sort_values(ascending=False))
        cossine_df.columns = ['recomendações']
        return cossine_df
    else:
        print(f"Erro: Filme '{filme_referencia}' não encontrado no dataset.")
        return None

def preparar_dados_conteudo(movies, tags, dados):
    """
    Prepara os dados para recomendação baseada em conteúdo
    
    Args:
        movies: DataFrame contendo informações dos filmes
        tags: DataFrame contendo tags dos filmes
        dados: DataFrame contendo dados adicionais
        
    Returns:
        DataFrame preparado para análise de conteúdo
    """
    print("Preparando dados para recomendação baseada em conteúdo...")
    
    # Converter movieId para string para garantir o merge correto
    movies['movieId'] = movies['movieId'].apply(lambda x: str(x))
    
    # Mesclar datasets para criar um único DataFrame enriquecido
    df2 = movies.merge(dados, left_on='title', right_on='Name', how='left')
    df2 = df2.merge(tags, left_on='movieId', right_on='movieId', how='left')
    
    # Usar gêneros como base para informações de conteúdo
    # Poderia ser expandido para incluir descrições e tags
    df2['Infos'] = df2['genres']  # + str(df2['Discription']) + df2['tag']
    
    print(f"Dataset baseado em conteúdo preparado com dimensões: {df2.shape}")
    return df2

def recomendacao_baseada_em_conteudo(df2, filme_referencia='Forrest Gump'):
    """
    Implementa o método de recomendação baseada em conteúdo usando TF-IDF e similaridade de cosseno
    
    Args:
        df2: DataFrame enriquecido com informações de conteúdo
        filme_referencia: Nome do filme para recomendar similares
        
    Returns:
        DataFrame com filmes similares ordenados
    """
    print(f"Calculando filmes similares a '{filme_referencia}' baseado em conteúdo...")
    
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
        print(f"Erro: Filme '{filme_referencia}' não encontrado no dataset.")
        return None

def mostrar_recomendacoes(recomendacoes, n=10, filme_referencia=None):
    """
    Exibe as recomendações de filmes
    
    Args:
        recomendacoes: DataFrame com as recomendações
        n: Número de recomendações a exibir
        filme_referencia: Nome do filme de referência
    """
    if recomendacoes is not None and not recomendacoes.empty:
        print(f"\nTop {n} recomendações para '{filme_referencia}':")
        print(recomendacoes.head(n))
    else:
        print("Não foi possível gerar recomendações.")

def main():
    """Função principal para executar o sistema de recomendação"""
    print("Sistema de Recomendação de Filmes 🎬")
    print("=" * 40)
    
    # Carregar dados
    movies, ratings, tags, dados = carregar_dados()
    
    # Filme para o qual queremos encontrar recomendações
    filme_referencia = 'Forrest Gump'
    
    print("\n1. MÉTODO DE FILTRAGEM COLABORATIVA")
    print("-" * 40)
    
    # Preparar dados e executar filtragem colaborativa
    movies_table = preparar_dados_colaborativos(movies, ratings)
    recomendacoes_colaborativas = filtragem_colaborativa(movies_table, filme_referencia)
    mostrar_recomendacoes(recomendacoes_colaborativas, 10, filme_referencia)
    
    if tags is not None and dados is not None:
        print("\n2. MÉTODO BASEADO EM CONTEÚDO")
        print("-" * 40)
        
        # Preparar dados e executar recomendação baseada em conteúdo
        df2 = preparar_dados_conteudo(movies, tags, dados)
        recomendacoes_conteudo = recomendacao_baseada_em_conteudo(df2, filme_referencia)
        mostrar_recomendacoes(recomendacoes_conteudo, 10, filme_referencia)
    
    print("\nProcesso de recomendação concluído!")

# Executar o script se for chamado diretamente
if __name__ == "__main__":
    main()