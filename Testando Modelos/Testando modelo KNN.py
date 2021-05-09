#!/usr/bin/env python
# coding: utf-8

# # Testando KNN

# ## **Chamando as bibliotecas e os dados**

# In[1]:


# Atualizando o scikit learn
#!pip install sklearn --upgrade


# In[2]:


import numpy as np
import pandas as pd

#Visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Divisão dos dados
from sklearn.model_selection import train_test_split

# Normalização
from sklearn.preprocessing import MinMaxScaler

# Redução de dimensionalidade
from sklearn.decomposition import PCA

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Grid Search
from sklearn.model_selection import GridSearchCV

# Métrica Perda em Log
from sklearn.metrics import log_loss

# Validação cruzada
from sklearn.model_selection import cross_validate


# In[3]:


dados_experimentos = pd.read_csv("https://github.com/FelipeN494/Desenvolvimento_medicamentos_Imersao_dados3_Alura/blob/main/Dados/dados_experimentos.zip?raw=true", compression = "zip")


# In[4]:


dados_resultados = pd.read_csv("https://github.com/FelipeN494/Desenvolvimento_medicamentos_Imersao_dados3_Alura/blob/main/Dados/dados_resultados.csv?raw=true")


# In[5]:


# Unificando os dados
dados_completos =  pd.merge(dados_experimentos, dados_resultados, on = "id")


# In[6]:


# Removendo Hífens
dados_completos.columns = dados_completos.columns.str.replace("-","")


# ## Redução de Dimensionalidade: PCA
# 
# O objetivo dessa parte é realizar o procedimento de formatação, separação e redução dos dados. Com isso, será possível oferecer dados que poderão ser utilizados pelo aprendizado de máquinas para fins de previsão. 
# 

# ### **Transformação das colunas tratamento e dose em valores numéricos.**
# 
# Antes de normalizar precisamos transformar as colunas tratamento e dose para formato numérico. Uma outra mudança antes necessária é modificar a coluna tempo: Nesta os valores 24, 48 e 72 horas serão representados por 0, 0.5 e 1 respectivamente.
# 

# In[7]:


# Transformando a coluna tratamento em dummies.
dados_completos.tratamento = pd.get_dummies(dados_completos.tratamento)

# Vamos ver os resultados.
# Podemos notar que os valores 0 e 1 representam "com_droga" e "com_controle" respectivamente.
dados_completos


# In[8]:


#Vamos repetir o mesmo procedimento para a coluna dose.
dados_completos.dose = pd.get_dummies(dados_completos.dose)

# O pandas vais transformar nossos dados para os valores 0 e 1, representando D1 e D2 respectivamente.
dados_completos["dose"].replace([0,1],[1,0], inplace = True)
#Vendo os resultados.
dados_completos


# In[9]:


dados_completos.iloc[:,871:]


# ### **Separação dos dados**
# 
# Aqui será feita a separação dos dados em três colunas: treino, validação e teste. O objetivo é garantir que os dados de treino e validação não contaminem os dados de teste (que servem para ver a eficiência do modelo quando confrontado com dados novos). Além disso, os dados de treinamento servem de padrão para transformar os demais.
# 

# In[10]:


#Separando os dados em x(entrada) e y(saída), é necessário remover a coluna droga também.
x = dados_completos.loc[:, "id":"c99"]
x = x.drop("droga", axis = 1)
y = dados_completos.loc[:, "5alpha_reductase_inhibitor":]

# Vamos separar os dados (usamos o random_state para garantir que a separação seja sempre a mesma com esses dados).

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = .2, random_state = 10)
x_treino, x_val, y_treino, y_val = train_test_split(x_treino,y_treino, test_size = .2, random_state = 10)


# In[11]:


x_treino


# ### **Normalização**
# Antes de realizar a transformação dos dados, é necessário normalizar estes. Isso porque a escala deles pode afetar a criação dos componentes princiais. Com a normalização os dados vão ficar em uma escala de 0 e 1. 
# 
# A normalização funciona extraindo de cada dado (X) o valor mínimo encontrado na sua coluna (Xmin) e dividide pela subtração do valor máximo pelo mínimo(Xmax - Xmin). Isto é:
# 
# $\ {X}_{normalizado} = \frac{{X}_{i} - {X}_{min}}{{X}_{max} - {X}_{min}} $ 
# 
# Uma mudança a parte deve ser feita na coluna tempo. Nesta os valores 24, 48 e 72 horas serão representados por 0, 0.5 e 1 respectivamente.
# 

# In[12]:


# Chamamos a função de normalização.
normalização = MinMaxScaler()

#Colocamos os valores de treino para normalizar. 
normalização.fit(x_treino.drop(["id"], axis = 1))

# Tranformamos os dados

normalização.transform(x_treino.drop(["id"], axis = 1))
normalização.transform(x_treino.drop(["id"], axis = 1))
normalização.transform(x_treino.drop(["id",], axis = 1))

# Visualizando os resultados
x_treino


# ### **Redução de dimensionalidade**
# 
# O objetivo dessa seção é transformar nossos dados de experimentos em menos colunas usando da Análise de Componentes Principais (Principal Componentes Analysis ou PCA). O PCA é explicado por Bruce e Bruce (2019, p. 259): 
# 
# 
# > "A ideia de PCA é combinar múltiplas variáveis preditoras numéricas em um conjunto menor de variáveis, que são combinações lineares ponderadas do conjunto original. O menor conteudo de variáveis, os *componentes principais*, 'explica' a maior parte da variabilidade do conjunto completo de variáveis, reduzindo a dimensão dos dados."
# 
# Em resumo, PCA é uma técnica estatística para diminuir a complexidade dos dados, mas preservando o máximo de informação importante. Com isso podemos obter um modelo com menos ruido, que não demore em demasia para ser treinado e testado (o que seria inviável se ele utilizasse todos os dados disponíveis) e que possa fazer previsões melhores que o modelo "burro".  

# In[13]:


# Vamos tentar reduzir de mais 800 colunas para 100 e ver a variância explicada
pca = PCA(n_components = 50, random_state= 2)

# Precisamos ajustar os dados de treino ao PCA
pca.fit(x_treino.drop("id", axis = 1))

# Vamos ver o quanto da variância observada o PCA explica.
pca.explained_variance_ratio_.sum()


# In[14]:


# Podemos visualizar também a variância explicada em um gráfico de linha.

plt.figure(figsize = (10,10))
sns.lineplot(x = range(1,51), y = pca.explained_variance_ratio_)
plt.title("Variância explicada por 50 componentes principais", fontsize = 14, weight = "bold")
plt.xlabel("PCA", fontsize = 12)
plt.ylabel("Variância explicada", fontsize = 12)


# In[15]:


# Precisamos transformar os nossos dados antes de treinar e testar o modelo.
# Começando pelo dados de treino
pca_xtreino = pca.transform(x_treino.drop("id", axis = 1))
x_treino = x_treino.filter(["id"])
x_treino = pd.concat([x_treino, pd.DataFrame(pca_xtreino, index = x_treino.index)], axis = 1, ignore_index= False)

# Passando para os dados de validação
pca_xval = pca.transform(x_val.drop("id", axis = 1))
x_val = x_val.filter(["id"])
x_val = pd.concat([x_val, pd.DataFrame(pca_xval, index = x_val.index)], axis = 1, ignore_index= False)

# Finalmente chegando nos dados de teste
pca_xteste = pca.transform(x_teste.drop("id", axis = 1))
x_teste = x_teste.filter(["id"])
x_teste = pd.concat([x_teste, pd.DataFrame(pca_xteste, index = x_teste.index)], axis = 1, ignore_index= False)


# ## Treinando e validando o KNN

# In[16]:


# Originalmente, tentou-se essa lista de hiperparâmetros, mas foi necessário testar menos hiperparâmetros em separado.
# Essa não é a melhor prática. Pode ser que os hiperparâmetros atuem de modo distinto em conjunto do que individualmente.
# Todavia, infelizmente, devemos utilizar um segundo melhor (a alternativa seria reduzir ainda mais oos dados, mas prefiro não sacrificar mais informação).

# hiperparametros = {"n_neighbors": [3,5,7],
#                    "weights": ["uniform", "distance"],
#                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
#                    "leaf_size": [10,20,30]}


# In[17]:


# Primeiro Grid Search.
# Escolhendo hiperparâmetros e testando.
knn = KNeighborsClassifier()
hiperparametros = {"n_neighbors": [3,5,7]}
GV = GridSearchCV(knn, hiperparametros)

# Vamos fazer o fit.
GV.fit(x_treino.drop("id", axis = 1), y_treino)

# Finalmente, vamos ver os resultados e os melhores hiperparâmetros.

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[18]:


# Segundo Grid Search.
# Vamos usar agoras os hiperparâmetros escolhidos do exercício anterior.
# Escolhendo hiperparâmetros e testando.
hiperparametros = {"n_neighbors": [5],
                    "weights": ["uniform", "distance"]}
GV = GridSearchCV(knn, hiperparametros)

# Vamos fazer o fit.
GV.fit(x_treino.drop("id", axis = 1), y_treino)

# Finalmente, vamos ver os resultados e os melhores hiperparâmetros.

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[19]:


# Terceiro Grid Search.
# Vamos usar agoras os hiperparâmetros escolhidos dos exercícios anteriores.
# Escolhendo hiperparâmetros e testando.
hiperparametros = {"n_neighbors": [5],
                   "weights": ["uniform"],
                   "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}
GV = GridSearchCV(knn, hiperparametros)

# Vamos fazer o fit.
GV.fit(x_treino.drop("id", axis = 1), y_treino)

# Finalmente, vamos ver os resultados e os melhores hiperparâmetros.

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[20]:


# Quarto Grid Search.
# Vamos usar agoras os hiperparâmetros escolhidos dos exercícios anteriores.
# Escolhendo hiperparâmetros e testando.
hiperparametros = {"n_neighbors": [5],
                   "weights": ["uniform"],
                   "algorithm": ["auto"],
                  "leaf_size": [10,20,30]}
GV = GridSearchCV(knn, hiperparametros)

# Vamos fazer o fit.
GV.fit(x_treino.drop("id", axis = 1), y_treino)

# Finalmente, vamos ver os resultados e os melhores hiperparâmetros.

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[21]:


# Dados para validação cruzada.
x_cv = pd.concat([x_treino, x_val], axis = 0)
y_cv = pd.concat([y_treino, y_val], axis = 0)


# In[22]:


# Fazendo CV para ver performance geral do "melhor" modelo.
knn = KNeighborsClassifier(algorithm = "auto", 
                           n_neighbors=5,
                           leaf_size = 10,
                           weights = "uniform")

CV = cross_validate(knn, 
                    x_cv.drop("id", axis = 1), 
                    y_cv, 
                    cv=5, 
                    return_train_score = True)


# In[23]:


# Vamos ver os resultados.
print("Notas de teste:", CV["test_score"], "\n", "Notas de treinamento:", CV["train_score"], "\n", "Notas de tempo de ajuste:", CV["score_time"])


# In[24]:


# Vamos ver então a perda em log.
log_loss(y_val, knn.fit(x_treino.drop("id", axis = 1), y_treino).predict(x_val.drop("id", axis = 1)))

