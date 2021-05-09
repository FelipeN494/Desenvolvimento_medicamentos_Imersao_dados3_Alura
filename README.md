# Desenvolvimento_medicamentos_Imersao_dados3_Alura
![image](https://user-images.githubusercontent.com/60203850/116947930-649a5b80-ac54-11eb-9035-fe14a0c02b10.png)

Autor: madartzgraphics. Fonte: https://pixabay.com/illustrations/dna-microscopic-cell-gene-helix-1903318/

# **Projeto: Prevendo Mecanismos de Ação** 

## Data de Início e Entrega

As aulas da terceira imersão de dados Alura foram entre 03 e 07 de Maio de 2021.
O documento foi escrito entre 06 e 09 de Maio de 2021, com data limite de entrega em  09 de Maio de 2021.

##  **Desenvolvimento de medicamentos, IA e aprendizado de máquinas**

A geração de medicamentos novos é um processo complexo e custoso, mas de grande valor social. Complexo porque existem diversas etapas da descoberta e análise do gene, escolha e teste de potenciais drogas, a testagem de múltiplas drogas em animais, depois testagem em voluntários humanos (saudáveis, depois doentes) seguida de testagem em uma população massiva e enfim ser aprovada pela agência pública de saúde responsável (ROWE,2020). Apesar dos custos médios do desenvolvimento de drogas ser controversos, pode-se dizer que eles atingem centenas de milhões de dólares (WOUTERS et. al., 2020). Ainda assim, o a geração de drogas novas é fundamental para os avanços da medicina e consequente melhorias na qualidade de vida humana.

Em vista das dificuldades e riscos no desenvolvimento de drogas, as empresas biofarmacêuticas buscam novas formas de aprimorar o processo; e aqui entra o aprendizado de máquinas. Companhias da indústria farmacêutica (como Pfizer), universidades e outros centros de pesquisa vem utilizando de IA para auxiliar no processo de compreensão de doenças e desenvolvimento de drogas (FLEMING, 2018). Uma das áreas em que o aprendizado de máquina e inteligência artificial vêm se mostrando mais promissores é a de descoberta de compostos que possam ser utilizados na criação de drogas, reduzindo o tempo de pesquisa e encontrando padrões muitas vezes imperceptíveis as pessoas (MULLARD, 2017).

## **Objetivo e dados**

Na linha desses desenvolvimentos, este projeto de conclusão de curso da Terceira edição da imersão de dados 03 da Alura tem por objetivo fornecer um modelo de machine learning para auxiliar no processo de descoberta de medicamentos. Para isso se utilizou de duas base de dados: uma contendo experimentos com compostos, nos quais se avalia a resposta de um conjunto de genes (se houve intensificação ou redução nas atividades de geração de proteina a partir do material genético) e da viabilidade (sobreviência) de certos de tipos de células; e um conjunto de dados de resultado que contém informações sobre os tipos de de Mecanismos de Ação (MoA) de cada experimento, indicando como atua cada um dos compostos segundo certa dose e tempo de aplicação. Mais especificamente, o objetivo consiste em prever qual (quais) Mecanismo(s) de Ação ativados em cada experimento (considerando que são mais de 200 mecanismos de ação). 

Trata-se de projeto de aprendizado de máquina do tipo supervisionado e de classificação múltipla (na medida em que cada composto pode ativar mais de um dos MoA); inspirado no desafio publicado pelo [Laboratory Innovation Science at Harvard](https://lish.harvard.edu/) no [Kaggle](https://www.kaggle.com/c/lish-moa). 

## Normalização e PCA

Após análise limpeza dos dados foi feita sua normalização e redução de complexidade via PCA.

## Modelo
## **Aprendizado de máquina**

Foi feita uma busca dos melhores modelos KNN e Árvore Extra em paralelo usando da função [GridSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). Em razão de sua melhor performance em reduzir a perda em Log e também em acurácia, optou-se por escolher a Árvore Extra. 

A Árvore Extra é um modelo que tira a média dos resultados de diversas Árvores de Decisão para fazer previsões. Uma Árvore de Decisão é um tipo de modelo estatístico que divide os dados com base em certas regras de decisão. Para mais detalhes recomendo que se veja a documentação das [Árvores de Decisão](https://scikit-learn.org/stable/modules/tree.html) [Árvores Extras](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) do Scikit Learn.  

## **Métricas**

Além de utilizarmos da métrica acurácia (total de acertos/total de previsões),em vista de manter coerência com o desafio postado no [Kaggle](https://www.kaggle.com/c/lish-moa/overview/evaluation), a avaliação do modelo se dará *também* pela métrica da 
perda em log (log loss) também conhecida como entropia binária cruzada (binary cross entopry); nosso objetivo é minimizar o valor do log.


## **Terminologia**

Para fins desse notebook os termos droga, composto e medicamento serão usados de maneira intercambiável. Todavia, destaca-se que eles, por claro, não se confundem. Na medida em que o composto é o elemento que causa reações no organismo, enquanto que a droga ou medicamento é a unidade que contém o composto e permite seu uso direto.

## Resultados e limitações

O modelo conseguiu obter pontuações de acurácia próximo a 15% acima do modelo padrão ("burro") além de diminuir a perda em log em aproximadamente 10%. Todavia, isso não necessariamente significa que ele deva ser empregado para fins práticos (isso demanda avaliação do pesquisador ou outro interessado). Por limitações de tempo (o desafio precisava ser entregue em 09 de Maio de 2021), não foi possível se explorar modelos mais complexos (Rede Neural, por exemplo) ou outras formas de diminuir a complexidade dos dados que não o PCA. Por isso recomenda-se que essas atividades sejam realizadas em pesquisas futuras. Além disso, outra limitação é que não foi possível tratar a tempo o desbalanceamento dos dados de saída; isso deve ser feito em pesquisaas futuras com o objetivo de aumentar a precisão do aprendizado de máquina.

## **Referências**

BROWNLEE, Jason. *How to Develop an Extra Trees Ensemble with Python*. 22 Abr. 2020. Disponível em:
<https://machinelearningmastery.com/extra-trees-ensemble-with-python/>. Acesso em: 08 Mai. 2021.

BRUCE, Peter; BRUCE, Andrew. *Estatística Prática para Cientista de Dados*. traduzido por Luciana Ferraz. Rio de Janeiro: Alta Books, 2019.

FLEMING, Nic. How artificial intelligence is changing drug discovery. Nature, v. 557, n. 7706, p. S55-S55, 2018. Disponível em: <https://www.nature.com/articles/d41586-018-05267-x>. Acesso em: 06 Mai. 2021.

MULLARD, Asher. The drug-maker's guide to the galaxy. Nature News, v. 549, n. 7673, p. 445, 2017. Disponível em: <https://www.nature.com/news/the-drug-maker-s-guide-to-the-galaxy-1.22683>. Acesso em: 06 Mai. 2021.

ROWE, Sebastian. *Modern Drug Discovery: Why is the drug development pipeline full of expensive failures?*.SITN. 21 Abr. 2020. Disponível em: <https://sitn.hms.harvard.edu/flash/2020/modern-drug-discovery-why-is-the-drug-development-pipeline-full-of-expensive-failures/>. Acesso em 06 Mai. 2021.

WOUTERS, Olivier J.; MCKEE, Martin; LUYTEN, Jeroen. Estimated research and development investment needed to bring a new medicine to market, 2009-2018. *Jama*, v. 323, n. 9, p. 844-853, 2020. Disponível em: <https://jamanetwork.com/journals/jama/fullarticle/2762311>. Acesso em: 06 Mai. 2021.

# Descrição das pastas

A pasta de dados e a descrição do desafio foram fornecidas pela Alura. Todo o resto desse repositório são de minha criação.

## Dados
Nesta pasta estão localizadas as bases de dados oferecidas pela Aluar para o projeto.

## Descrição do Desafio

Nesta pasta está a descrição do desafio final do projeto de imersão de dados 3.

## Testando modelos.

Essa pasta contém a pesquisa dos melhores hiperparâmetros utilizando GridSeatch() e a validação cruzada dos modelos KNN, Árvore Extra e também o Modelo Burro ("Dummy").

## Resultados
Essa pasta contém o notebook com o projeto completo.

# Especificações técnicas

Os notebooks foram criados e rodados em um computar com processador Intel(R) Core(TM) i5-8400 CPU @ 2.80 GHz; com 8 GB de memória instalada; sistema operacional 64 bits. O notebook principal foi rodado no google colaboratory (Colab) e os de treinamento e validação dos modelos no Jupyter (Anaconda) v. 6.3.0.
