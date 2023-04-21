# Pacotes
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

# Coleta os dados através da URL
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length","sepal_width", "petal_length", "petal_width", "class"] # Nome das colunas
dataset = pd.read_csv(url, names=attributes) 
dataset.columns = attributes

print(dataset.shape) # Mostra a quantidade de instâncias (linhas) e atributos (colunas) respectivamente
print(dataset.head(20)) # Mostra um dataframe com as primeiras 20 instâncias
print(dataset.describe()) # Mostra um resumo (contagem, média, valores mínimos e máximos e percentuais) de cada atributo
print(dataset.groupby('class').size()) # Mostra o número de instâncias (contagem absoluta) que pertecem a cada "class" 

# ===== Gráficos Univariados =====

# Gráficos de caixas e bigodes
dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)

# Histograma
dataset.hist()

# ===== Gráfico Multivariado =====

# Matriz do gráfico de dispersão
scatter_matrix(dataset)

# ================================

# Conjunto de dados de validação dividido 
array = dataset.valuesx 
x = array[:,0:4]
y = array[:,4]
validation_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed)

# Opções de teste e métrica de avaliação
scoring = "accuracy"
