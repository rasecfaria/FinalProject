#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recomendador Interativo de Filmes
================================
Script interativo que permite ao usuário escolher um filme 
e receber recomendações baseadas nos dois métodos implementados.
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from output.sistema_recomendacao import (carregar_dados, preparar_dados_colaborativos, 
                                  filtragem_colaborativa, preparar_dados_conteudo,
                                  recomendacao_baseada_em_conteudo, mostrar_recomendacoes)

def listar_filmes_populares(movies, ratings, n=20):
    """
    Lista os filmes mais populares baseado no número de avaliações
    
    Args:
        movies: DataFrame com informações dos filmes
        ratings: DataFrame com avaliações
        n: Número de filmes a mostrar
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
    
    print(f"\nTop {n} filmes mais populares:")
    for i, (_, row) in enumerate(filmes_populares.head(n).iterrows(), 1):
        print(f"{i}. {row['title']} ({row['num_avaliacoes']} avaliações)")
    
    return filmes_populares

def pesquisar_filme(movies, termo_pesquisa):
    """
    Pesquisa filmes por termo no título
    
    Args:
        movies: DataFrame com informações dos filmes
        termo_pesquisa: Termo para pesquisar nos títulos
    
    Returns:
        DataFrame com resultados da pesquisa
    """
    try:
        # Pesquisar de forma case-insensitive e com tratamento para valores NA
        resultados = movies[movies['title'].str.contains(termo_pesquisa, case=False, na=False)]
        
        if resultados.empty:
            print(f"Nenhum filme encontrado contendo '{termo_pesquisa}'")
            return None
        
        print(f"\nFilmes encontrados com '{termo_pesquisa}' ({len(resultados)} resultados):")
        for i, (_, row) in enumerate(resultados.iterrows(), 1):
            print(f"{i}. {row['title']}")
        
        return resultados
    except Exception as e:
        print(f"Erro ao pesquisar filmes: {e}")
        print("Tente outro termo de pesquisa sem caracteres especiais.")
        return None

def main():
    """Função principal para executar o recomendador interativo"""
    print("Recomendador Interativo de Filmes 🎬")
    print("=" * 40)
    
    # Carregar dados
    movies, ratings, tags, dados = carregar_dados()
    
    # Preparar dados para recomendação
    movies_table = preparar_dados_colaborativos(movies, ratings)
    
    if tags is not None and dados is not None:
        df2 = preparar_dados_conteudo(movies, tags, dados)
        metodo_conteudo_disponivel = True
    else:
        metodo_conteudo_disponivel = False
        df2 = None  # Garantir que df2 esteja definido mesmo quando não disponível
    
    while True:
        print("\nOpções:")
        print("1. Ver filmes populares")
        print("2. Pesquisar filme")
        print("3. Escolher filme por título completo")
        print("4. Sair")
        
        opcao = input("\nEscolha uma opção (1-4): ")
        
        if opcao == '1':
            filmes_populares = listar_filmes_populares(movies, ratings)
            indice = input("\nEscolha o número do filme para recomendações (ou 0 para voltar): ")
            
            try:
                indice = int(indice)
                if 0 < indice <= len(filmes_populares):
                    filme_escolhido = filmes_populares.iloc[indice-1]['title']
                    mostrar_recomendacoes_para_filme(filme_escolhido, movies_table, df2, metodo_conteudo_disponivel)
            except ValueError:
                print("Entrada inválida.")
            
        elif opcao == '2':
            termo = input("Digite parte do título do filme: ")
            resultados = pesquisar_filme(movies, termo)
            
            if resultados is not None and not resultados.empty:
                indice = input("\nEscolha o número do filme para recomendações (ou 0 para voltar): ")
                try:
                    indice = int(indice)
                    if 0 < indice <= len(resultados):
                        filme_escolhido = resultados.iloc[indice-1]['title']
                        mostrar_recomendacoes_para_filme(filme_escolhido, movies_table, df2, metodo_conteudo_disponivel)
                except ValueError:
                    print("Entrada inválida.")
            
        elif opcao == '3':
            filme_escolhido = input("Digite o título exato do filme: ")
            mostrar_recomendacoes_para_filme(filme_escolhido, movies_table, df2, metodo_conteudo_disponivel)
            
        elif opcao == '4':
            print("Obrigado por usar o Recomendador de Filmes!")
            break
            
        else:
            print("Opção inválida. Por favor, escolha novamente.")

def mostrar_recomendacoes_para_filme(filme, movies_table, df2, metodo_conteudo_disponivel):
    """
    Exibe recomendações para um filme específico usando ambos os métodos
    
    Args:
        filme: Nome do filme
        movies_table: Tabela pivot para filtragem colaborativa
        df2: DataFrame preparado para recomendação baseada em conteúdo
        metodo_conteudo_disponivel: Booleano indicando se o método está disponível
    """
    print(f"\nRecomendações para: {filme}")
    print("-" * 40)
    
    print("\n1. MÉTODO DE FILTRAGEM COLABORATIVA:")
    recomendacoes_colaborativas = filtragem_colaborativa(movies_table, filme)
    mostrar_recomendacoes(recomendacoes_colaborativas, 10, filme)
    
    if metodo_conteudo_disponivel and df2 is not None:
        print("\n2. MÉTODO BASEADO EM CONTEÚDO:")
        recomendacoes_conteudo = recomendacao_baseada_em_conteudo(df2, filme)
        mostrar_recomendacoes(recomendacoes_conteudo, 10, filme)

if __name__ == "__main__":
    main()