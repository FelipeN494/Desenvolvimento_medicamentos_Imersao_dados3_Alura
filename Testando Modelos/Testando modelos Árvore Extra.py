#!/usr/bin/env python
# coding: utf-8

# # Testando o modelo Árvore Extra
# 
# Nesse notebook primeiro se repete a formatção dos dados, seguida da testagem do modelo de Árvore Extra para classificação.

# ## **Chamando as bibliotecas e os dados**

# In[1]:


# Atualizando o scikit learn
# !pip install sklearn --upgrade


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

# Árvore Extra
from sklearn.ensemble import ExtraTreesClassifier

# Grid Search
from sklearn.model_selection import GridSearchCV

# Métrica Perda em Log
from sklearn.metrics import log_loss

# Validação cruzada
from sklearn.model_selection import cross_validate

# Shuffle Split
from sklearn.model_selection import ShuffleSplit


# In[3]:


dados_experimentos = pd.read_csv("https://github.com/FelipeN494/Desenvolvimento_medicamentos_Imersao_dados3_Alura/blob/main/Dados/dados_experimentos.zip?raw=true", compression = "zip")


# In[4]:


dados_resultados = pd.read_csv("https://github.com/FelipeN494/Desenvolvimento_medicamentos_Imersao_dados3_Alura/blob/main/Dados/dados_resultados.csv?raw=true")


# In[5]:


# Unificando os dados
dados_completos =  pd.merge(dados_experimentos, dados_resultados, on = "id")


# In[6]:


# Removendo hífens
dados_completos =  pd.merge(dados_experimentos, dados_resultados, on = "id")


# In[7]:


# Removendo Hífens
dados_completos.columns = dados_completos.columns.str.replace("-","")


# ## **Redução de Dimensionalidade: PCA**
# 
# O objetivo dessa parte é realizar o procedimento de formatação, separação e redução dos dados. Com isso, será possível oferecer dados que poderão ser utilizados pelo aprendizado de máquinas para fins de previsão. 
# 

# ### **Transformação das colunas tratamento e dose em valores numéricos.**
# 
# Antes de normalizar precisamos transformar as colunas tratamento e dose para formato numérico. Uma outra mudança antes necessária é modificar a coluna tempo: Nesta os valores 24, 48 e 72 horas serão representados por 0, 0.5 e 1 respectivamente.
# 

# In[8]:


# Transformando a coluna tratamento em dummies.
dados_completos.tratamento = pd.get_dummies(dados_completos.tratamento)

# Vamos ver os resultados.
# Podemos notar que os valores 0 e 1 representam "com_droga" e "com_controle" respectivamente.
dados_completos


# In[9]:


#Vamos repetir o mesmo procedimento para a coluna dose.
dados_completos.dose = pd.get_dummies(dados_completos.dose)

# O pandas vais transformar nossos dados para os valores 0 e 1, representando D1 e D2 respectivamente.
dados_completos["dose"].replace([0,1],[1,0], inplace = True)
#Vendo os resultados.
dados_completos


# In[10]:


dados_completos.iloc[:,871:]


# ### **Separação dos dados**
# 
# Aqui será feita a separação dos dados em três colunas: treino, validação e teste. O objetivo é garantir que os dados de treino e validação não contaminem os dados de teste (que servem para ver a eficiência do modelo quando confrontado com dados novos). Além disso, os dados de treinamento servem de padrão para transformar os demais.
# 

# In[11]:


#Separando os dados em x(entrada) e y(saída), é necessário remover a coluna droga também.
x = dados_completos.loc[:, "id":"c99"]
x = x.drop("droga", axis = 1)
y = dados_completos.loc[:, "5alpha_reductase_inhibitor":]

# Vamos separar os dados (usamos o random_state para garantir que a separação seja sempre a mesma com esses dados).

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = .2, random_state = 10)
x_treino, x_val, y_treino, y_val = train_test_split(x_treino,y_treino, test_size = .2, random_state = 10)


# In[12]:


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

# In[13]:


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

# In[14]:


# Vamos tentar reduzir de mais 800 colunas para 100 e ver a variância explicada
pca = PCA(n_components = 50, random_state= 2)

# Precisamos ajustar os dados de treino ao PCA
pca.fit(x_treino.drop("id", axis = 1))

# Vamos ver o quanto da variância observada o PCA explica.
pca.explained_variance_ratio_.sum()


# In[15]:


# Podemos visualizar também a variância explicada em um gráfico de linha.

plt.figure(figsize = (10,10))
sns.lineplot(x = range(1,51), y = pca.explained_variance_ratio_)
plt.title("Variância explicada por 50 componentes principais", fontsize = 14, weight = "bold")
plt.xlabel("PCA", fontsize = 12)
plt.ylabel("Variância explicada", fontsize = 12)


# In[16]:


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


# ## **Testando e validando a Árvore Extra**

# A escolha dos hiperparâmetros para serem "afinados" (tune) foi inspirada no texto:
# 
# BROWNLEE, Jason. *How to Develop an Extra Trees Ensemble with Python*. 22 Abr. 2020. Disponível em:
# <https://machinelearningmastery.com/extra-trees-ensemble-with-python/>. Acesso em: 08 Mai. 2021.

# In[17]:


# Idealmente a validação cruzada iria testar todos os hiperparâmetros.
# Infelizmente, em razão da quantidade dos dados, isso é computacionalmente inviável.
# Por isso, será necessário testar os hiperparâmetros separadamente.
# De qualquer modo, aqui fica a lista completa de hiperparâmetros.
AExtra = ExtraTreesClassifier()

# hiperparametros = {"n_estimators": [25,50,100, 200],
#                    "max_features": [2, 5, 7, 10],
#                    "min_samples_split" :[2, 5, 7, 10]}


# In[18]:


# Primeiro Grid Search.
hiperparametros = {"n_estimators": [25,50]}

GV = GridSearchCV(AExtra, hiperparametros)

GV.fit(x_treino.drop("id", axis = 1), y_treino)

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[19]:


# Segundo Grid Search.
hiperparametros = {"n_estimators": [25],
                  "max_features": [2, 5, 7, 10]}

GV = GridSearchCV(AExtra, hiperparametros)

GV.fit(x_treino.drop("id", axis = 1), y_treino)

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[20]:


# Terceiro Grid Search.
hiperparametros = {"n_estimators": [25],
                  "max_features": [10],
                  "min_samples_split" :[2, 5, 7, 10]}

GV = GridSearchCV(AExtra, hiperparametros)

GV.fit(x_treino.drop("id", axis = 1), y_treino)

print("Resultados:", GV.cv_results_, "\n", "Melhores hiperparâmetros:", GV.best_params_)


# In[21]:


# Dados para validação cruzada.
x_cv = pd.concat([x_treino, x_val], axis = 0)
y_cv = pd.concat([y_treino, y_val], axis = 0)


# In[22]:


# Fazendo CV para ver performance geral do "melhor" modelo.
AExtra = ExtraTreesClassifier(n_estimators = 25,
                  max_features = 10,
                  min_samples_split = 2)

CV = cross_validate(AExtra, 
                    x_cv.drop("id", axis = 1), 
                    y_cv, 
                    cv=5, 
                    return_train_score = True)


# In[23]:


# Vejamos os resultados.
print("Notas de teste:", CV["test_score"], "\n", "Notas de treinamento:", CV["train_score"], "\n", "Notas de tempo de ajuste:", CV["score_time"])


# In[24]:


# O modelo parece apresentar overfit.
# Vamos tentar reduzir isso utilizando do Shufle Split.
SSplit = ShuffleSplit(n_splits=5, test_size=.2, random_state= 10)


# In[25]:


# Testando
AExtra = ExtraTreesClassifier(n_estimators = 20,
                  max_features = 8,
                  min_samples_split = 4)

CV = cross_validate(AExtra, 
                    x_cv.drop("id", axis = 1), 
                    y_cv, 
                    cv=SSplit, 
                    return_train_score = True)
# Vejamos os resultados.
# Aparentemente, o modelo ainda tende a overfit com dados de treinamento.
print("Notas de teste:", CV["test_score"], "\n", "Notas de treinamento:", CV["train_score"], "\n", "Notas de tempo de ajuste:", CV["score_time"])


# In[26]:


# Vamos tentar reduzir isso mais modificando os hiperparâmetros.
# Testando
AExtra = ExtraTreesClassifier(n_estimators = 10,
                  max_features = 4,
                  min_samples_split = 8)

CV = cross_validate(AExtra, 
                    x_cv.drop("id", axis = 1), 
                    y_cv, 
                    cv=SSplit, 
                    return_train_score = True)
# Vejamos os resultados.
# Conseguimos diminuir o overfit com dados de treinamento e sem afetar os dados de teste.
print("Notas de teste:", CV["test_score"], "\n", "Notas de treinamento:", CV["train_score"], "\n", "Notas de tempo de ajuste:", CV["score_time"])


# In[27]:


# Vamos ver então a perda em log.
log_loss(y_val, AExtra.fit(x_treino.drop("id", axis = 1), y_treino).predict(x_val.drop("id", axis = 1)))

