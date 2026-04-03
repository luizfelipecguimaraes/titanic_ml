# 🚢 Previsão de Sobrevivência do Titanic com Machine Learning
by: Felipe Guimarães

Este projeto é um modelo de introdução ao Machine Learning em Python, desenvolvido para analisar os dados de passageiros do RMS Titanic e prever a probabilidade de sobrevivência com base em características como idade, classe, sexo e tarifa.

## 🎯 Objetivo
Realizar a limpeza, manipulação, análise exploratória (EDA) e treinamento de um modelo de Inteligência Artificial capaz de classificar os passageiros em "Sobreviveu" ou "Não Sobreviveu".

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Ambiente:** Spyder (Anaconda)
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização de Dados:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`)

## 📊 Estrutura do Projeto
O projeto foi dividido em três grandes etapas:

1. **Tratamento e Limpeza de Dados:** * Remoção de colunas com alta taxa de valores nulos (ex: `nivel_cabine`).
   * Substituição de valores faltantes na coluna `idade` utilizando a mediana estatística, para evitar a perda de dados cruciais.
2. **Feature Engineering:**
   * Criação da variável `tamanho_familia` (somando irmãos, cônjuges, pais e filhos a bordo).
   * Transformação de variáveis categóricas (texto) em valores numéricos utilizando a técnica de *One-Hot Encoding*.
3. **Machine Learning:**
   * Divisão dos dados em treino (80%) e teste (20%).
   * Treinamento utilizando o algoritmo **Random Forest**, com ajustes de hiperparâmetros (como `max_depth` e `n_estimators`) para evitar *overfitting*.

## 📈 Resultados Alcançados
Após o treinamento, o modelo foi submetido à base de testes e avaliado através de métricas de classificação.

* **Acurácia Geral:** ~81% (O modelo acerta 8 em cada 10 previsões).
* Os recursos que o modelo considerou mais importantes para determinar a sobrevivência foram:
  1. Sexo (Mulheres tiveram prioridade)
  2. Valor da Tarifa (Ligado diretamente à classe socioeconômica)
  3. Classe do Navio (Passageiros da 1ª classe tiveram maiores chances)

## 📁 Como executar este projeto
1. Clone este repositório para a sua máquina local.
2. Certifique-se de ter o Python e as bibliotecas listadas instaladas (recomenda-se o uso do Anaconda).
3. Execute o script `titanic_ml.py`.
