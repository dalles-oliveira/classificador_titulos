import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
import string
import nltk
from joblib import Parallel, delayed

# Verifica se o punkt do NLTK está instalado para português
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Carregar modelo CBOW e dados
modelo_cbow = KeyedVectors.load_word2vec_format("modelos/cbow_s300.txt")
artigo_treino = pd.read_csv("data/treino.csv")
artigo_teste = pd.read_csv("data/teste.csv")

# Função para tokenização
def tokenizador(texto):
    texto = texto.lower()
    lista_alfanumerico = []
    for token_valido in nltk.word_tokenize(texto):
        if token_valido in string.punctuation: continue
        lista_alfanumerico.append(token_valido)
    return lista_alfanumerico

# Função para combinação de vetores usando CBOW
def combinacao_de_vetores_por_soma(palavras_numeros):
    vetor_resultante = np.zeros(300)
    for pn in palavras_numeros:
        try:
            vetor_resultante += modelo_cbow.get_vector(pn)
        except KeyError:
            if pn.isnumeric():
                pn = "0"*len(pn)
                vetor_resultante += modelo_cbow.get_vector(pn)
            else:
                vetor_resultante += modelo_cbow.get_vector("unknown")
    return vetor_resultante

# Função para criar vetor de texto usando CBOW em paralelo
def criar_vetor_texto_cbow_paralelo(palavras_numeros):
    resultados = Parallel(n_jobs=-1)(delayed(modelo_cbow.get_vector)(pn) for pn in palavras_numeros)
    vetor_resultante = np.sum(resultados, axis=0)
    return vetor_resultante

# Interface Streamlit
st.title('Classificação de Títulos de Notícias')
titulo_noticia = st.text_input('Digite o título da notícia:')

if st.button('Classificar'):
    palavras_numeros = tokenizador(titulo_noticia)

    # Criar vetor de texto usando CBOW em paralelo
    vetor_texto_cbow = criar_vetor_texto_cbow_paralelo(palavras_numeros)

    # Reshape para compatibilidade com o modelo de regressão logística
    vetor_texto_cbow = vetor_texto_cbow.reshape(1, -1)

    # Modelo de classificação usando CBOW
    LR_cbow = LogisticRegression(max_iter=200)
    LR_cbow.fit(matriz_vetores_treino, artigo_treino.category)

    # Predição da categoria
    categoria_predita_cbow = LR_cbow.predict(vetor_texto_cbow)

    st.write('**Modelo CBOW:**', categoria_predita_cbow[0])
