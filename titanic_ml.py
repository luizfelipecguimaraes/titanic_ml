#%% Importando bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Importando

titanic = pd.read_excel('titanic.xlsx')
titanic.info()

#%% Tratamento e Manipulação

classe = titanic['classe']
selecao_var = titanic[['sobreviveu', 'classe', 'sexo']]

selecao_var = selecao_var.drop(columns=['classe'])

#%% Selecionando observações

# Adultos do sexo masculino 
masc_adultos = (titanic[(titanic['sexo'] == 'masculino') & 
                        (titanic['idade'] >= 18)])

# Mulheres ou crianças
mulheres_criancas = (titanic[(titanic['sexo'] == 'feminino') |
                             (titanic['idade'] < 18)])

# Pessoas sozinhas (sem parentes)
sozinhos = (titanic[(titanic['irmaos_conjuges'] == 0) &
                    (titanic['pais_filhos'] == 0)])

# Pessoas que pagaram mais de $100 pela tarifa e embarcaram em Cherbourg
tarifas_embarque = (titanic[(titanic['valor_tarifa'] > 100) &
                            (titanic['embarque'] == 'Cherbourg')])

#%% Limpeza de dados

titanic_limpo = titanic.drop(columns=['nivel_cabine']).dropna()

#%% Estatísticas descritivas: tabela de frequências

sozinhos['sobreviveu'].value_counts(normalize=True)
masc_adultos['sobreviveu'].value_counts(normalize=True)

#%% Estatísticas descritivas: variável quantitativa

# Contagem de observações
titanic['idade'].count()

# Média aritmética
titanic['idade'].mean()

# Mediana
titanic['idade'].median()

# Máximo
titanic['idade'].max()

# Mínimo
titanic['idade'].min()

# Quartis e Percentis
titanic['idade'].quantile(q = 0.25) # Primeiro Quartil
titanic['idade'].quantile(q = 0.75) # Terceiro Quartil

# Variância
titanic['idade'].var()

# Desvio padrão
titanic['idade'].std()

# Tabela de descritivas
titanic['idade'].describe()
titanic[['idade', 'valor_tarifa']].describe()
titanic.describe()

#%% Estatísticas descritivas: agrupadas por meio de critério qualitativo

# Valor médio da tarifa por classe
titanic[['classe', 'valor_tarifa']].groupby(by=['classe']).mean()

# Percentual de sobreviventes por sexo
titanic[['sobreviveu', 'sexo']].groupby('sexo')['sobreviveu'].value_counts(normalize=True)

# Percentual de sobreviventes por classe
titanic[['sobreviveu', 'classe']].groupby('classe')['sobreviveu'].value_counts(normalize=True)

#%% Data Visualization

# Gráfico de barras para contagem de sobreviventes por sexo
plt.figure(figsize=(15,9), dpi=600)
ax = sns.countplot(data=titanic, x='sobreviveu', hue='sexo', palette='viridis', legend=True)
for container in ax.containers: ax.bar_label(container, fontsize=12)
plt.title('Sobreviventes por sexo',fontsize=20)
plt.xlabel('Sobreviveu?',fontsize=15)
plt.ylabel('Contagem',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Gráfico de barras geral do valor médio da tarifa em cada local de embarque 
dados = titanic[['embarque', 'valor_tarifa']].groupby(by=['embarque']).mean()
plt.figure(figsize=(15,9), dpi=600)
ax1 = sns.barplot(data=dados, x=dados.index, y='valor_tarifa', hue=dados.index, palette='bright')
for container in ax1.containers: ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=12)
plt.title("Tarifa Média por Cidade de Embarque",fontsize=20)
plt.xlabel('Cidade de Embarque',fontsize=15)
plt.ylabel('Valor',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Histograma da idade
plt.figure(figsize=(15,9), dpi=600)
sns.histplot(data=titanic, x='idade', bins=range(0,85,5), kde=True, color='lightgreen')
plt.xlabel('Idade',fontsize=15)
plt.ylabel('Frequência',fontsize=15)
plt.xticks(ticks=np.arange(0,85,5), fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Gráfico de dispersão de pontos
#x = Idade; y = Valor da Tarifa
plt.figure(figsize=(15,9), dpi=600)
sns.scatterplot(data=titanic[titanic['valor_tarifa']<100], x='idade', y='valor_tarifa', 
                hue='classe', hue_order=['primeira', 'segunda', 'terceira'])
plt.title('Dispersão da Idade pelo Valor da Tarifa',fontsize=20)
plt.xlabel('Idade',fontsize=15)
plt.ylabel('Valor da Tarifa',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Gráfico de linhas de Tarifa por Classe
dados_linha = titanic[['classe', 'valor_tarifa']].groupby(by=['classe']).mean()
plt.figure(figsize=(15,9), dpi=600)
sns.lineplot(data=dados_linha, x=dados_linha.index, y='valor_tarifa', marker='o', linewidth=3, color='purple')
plt.title('Valor por Classe',fontsize=20)
plt.xlabel('Classes',fontsize=15)
plt.ylabel('Tarifa Média',fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Boxplot - idade
plt.figure(figsize=(15,9), dpi=600)
sns.boxplot(y=titanic['idade'], color='orange')

minimo = titanic['idade'].min()
q1 = titanic['idade'].quantile(q = 0.25)
q2 = titanic['idade'].median()
q3 = titanic['idade'].quantile(q = 0.75)
maximo = titanic['idade'].max()

plt.text(0, minimo, f'Mín = {minimo}', ha='center', va='top', fontweight='bold')
plt.text(0, q1, f'Q1 = {q1:.1f}', ha='center', va='top', fontweight='bold')
plt.text(0, q2, f'Q2 = {q2:.1f}', ha='center', va='center', fontweight='bold')
plt.text(0, q3, f'Q3 = {q3:.1f}', ha='center', va='bottom', fontweight='bold')
plt.text(0, maximo, f'Máx = {maximo}', ha='center', va='bottom', fontweight='bold')
plt.title('Boxplot da Idade',fontsize=20)
plt.show()

#%% Machine Learning - Preparação dos dados

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_ml = titanic_limpo.copy()

# Alvo
df_ml['sobreviveu'] = df_ml['sobreviveu'].map({'sim': 1, 'nao': 0})

# One-Hot Encoding
df_ml = pd.get_dummies(df_ml, columns=['classe', 'sexo', 'embarque'], drop_first=True)

#%% Machine Learning - Treinamento e Teste

X = df_ml.drop(columns=['sobreviveu']) # Tudo que vamos usar para prever
y = df_ml['sobreviveu']                # O que queremos prever

# Treino 80% e Teste 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utilizando o modelo RandomForestClassifier
modelo = RandomForestClassifier(random_state=42)

modelo.fit(X_train, y_train)

#%% Machine Learning - Avaliação do Modelo

# Ver se o modelo está acertando os dados de teste
previsoes = modelo.predict(X_test)

# Avaliando a performance
acuracia = accuracy_score(y_test, previsoes)
print(f"Acurácia do Modelo: {acuracia * 100:.2f}%")
print("\nRelatório de Classificação:\n", classification_report(y_test, previsoes))

# Matriz de Confusão
plt.figure(figsize=(6,4), dpi=300)
sns.heatmap(confusion_matrix(y_test, previsoes), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão do Modelo')
plt.xlabel('Previsão do Modelo (0=Morreu, 1=Sobreviveu)')
plt.ylabel('Realidade (0=Morreu, 1=Sobreviveu)')
plt.show()


'''
Os resultados desse treinamento para identificação dos casos de sobreviventes foram:
Acurácia do Modelo: 78.32%
Relatório de Classificação:
               precision    recall  f1-score   support

           0       0.79      0.84      0.81        80
           1       0.78      0.71      0.74        63

    accuracy                           0.78       143
   macro avg       0.78      0.78      0.78       143
weighted avg       0.78      0.78      0.78       143
'''

#%% Tentando melhorar a acurácia a partir de outra coluna chave.

'''Utilizando o dataset original e ao invés de remover todas as linhas com algo vazio, 
apenas tratar melhor os dados e preencher eles através de algum calculo, como a mediana.'''
df_melhorado = titanic.drop(columns=['nivel_cabine']).copy()

mediana_idade = df_melhorado['idade'].median()
df_melhorado['idade'] = df_melhorado['idade'].fillna(mediana_idade)
df_melhorado = df_melhorado.dropna()

# Feature Engineering
# Criando a coluna Tamanho da Família (+1 representa a própria pessoa)
df_melhorado['tamanho_familia'] = df_melhorado['irmaos_conjuges'] + df_melhorado['pais_filhos'] + 1
df_melhorado['sozinho'] = (df_melhorado['tamanho_familia'] == 1).astype(int)

# Preparação
df_melhorado['sobreviveu'] = df_melhorado['sobreviveu'].map({'sim': 1, 'nao': 0})
df_melhorado = pd.get_dummies(df_melhorado, columns=['classe', 'sexo', 'embarque'], drop_first=True)

X_melhor = df_melhorado.drop(columns=['sobreviveu'])
y_melhor = df_melhorado['sobreviveu']

# Separando Treino e Teste
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_melhor, y_melhor, test_size=0.2, random_state=42)

# Tuning
# Profundidade máxima sendo 5 com 100 árvores
modelo_melhorado = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

# Treinando
modelo_melhorado.fit(X_train_m, y_train_m)

# Avaliando
previsoes_m = modelo_melhorado.predict(X_test_m)
acuracia_m = accuracy_score(y_test_m, previsoes_m)

print(f"Antiga Acurácia: {acuracia * 100:.2f}%")
print(f"Nova Acurácia Alcançada: {acuracia_m * 100:.2f}%")

#%% Avaliando o modelo

print("--- RELATÓRIO DE CLASSIFICAÇÃO ---")
print(classification_report(y_test_m, previsoes_m))

# Matriz de Confusão
plt.figure(figsize=(8, 5), dpi=600)
matriz = confusion_matrix(y_test_m, previsoes_m)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto: Morreu (0)', 'Previsto: Sobreviveu (1)'], 
            yticklabels=['Real: Morreu (0)', 'Real: Sobreviveu (1)'])
plt.title('Matriz de Confusão', fontsize=16)
plt.yticks(rotation=0)
plt.show()

# Quais colunas o Random Forest achou mais importantes para tomar a decisão
importancias = pd.Series(modelo_melhorado.feature_importances_, index=X_melhor.columns)
importancias = importancias.sort_values(ascending=False)

plt.figure(figsize=(10,6), dpi=300)
sns.barplot(x=importancias, y=importancias.index, palette='magma')
plt.title('Variáveis mais importantes para prever a sobrevivência')
plt.xlabel('Grau de Importância')
plt.ylabel('Variáveis')
plt.show()

'''
Os resultados do novo treinamento foram:
Acurácia do Modelo: 81.46%
Relatório de Classificação:
                  precision    recall  f1-score   support

            0       0.82      0.88      0.85       109
            1       0.79      0.70      0.74        69

    accuracy                           0.81       178
   macro avg       0.80      0.79      0.79       178
weighted avg       0.81      0.81      0.81       178
'''
