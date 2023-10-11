# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 20:08:21 2023

@author: alsouza
"""

import pandas as pd


arquivo = pd.read_csv('weatherAUS.csv')

#------------------- Análise Eploratória dos dados ------------------------- 

""" Obtendo cabeçalho"""
print(arquivo.head())

""" Obtendo o número de linhas e colunas 142.193 linhas cento e quarenta duas mi e cento e noventa e três linhas 24 colunas """
print(arquivo.shape)

""" Convertendo dados de coluna que são classes para valor 0 ou 1 no caso RainToday e RainTomorrow"""

arquivo['RainToday'].replace({'No':0, 'Yes':1},inplace = True)
arquivo['RainTomorrow'].replace({'No':0, 'Yes':1},inplace = True)


### Importando Biblioteca para fazer um gráfico e gerando gráfico ############ 
import matplotlib.pyplot as plt

#definindo o tamanho do gráfico
fig = plt.figure(figsize = (8,5))


#gerando gráfico normalize = TRue significa escala de [0.0, 1.0] sendo a area do retangulo igual 1
arquivo.RainTomorrow.value_counts(normalize = True ).plot(kind='bar', color= ['skyblue','navy'], alpha = 1, rot=0)

#titulo do Gráfico
plt.title('RainTomorrow Indicator No(0) and Yes(1) in the Imbalanced Dataset')
#mostrando o gráfico

## Primeiro gRáfico ######
plt.show()


""" É observado que existe uma descrepância em relação aos dados da classe pois os do Tipo 0 são muitos e 1 poucos
concluímos que o dataset esta desbalanceado, e precisamos balancea-ló""" 

# Utilizando a biblioteca Sklearn e pegando o metodo reseample para equalizar a classe desbalanceada

from sklearn.utils import resample
import seaborn as sns


zeros= arquivo[arquivo.RainTomorrow == 0]
ums= arquivo[arquivo.RainTomorrow == 1]

oversampleando =  resample(ums, replace=True, n_samples=len(zeros), random_state=123)

feito= pd.concat([zeros, oversampleando])

fig = plt.figure(figsize = (8,5))

feito.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')

##### Segundo Gráfico ####

plt.show()


#Agora que a classe esta balanceada Vamos tratar os DADOS  FALTANTES

#Primeiramente apresentando graficamente os dados Faltantes

sns.heatmap(oversampleando.isnull(), cbar=False, cmap='PuBu')

""" Foi visto então que as colunas Sunshine, Evaporation, Cloud9am e Cloud 3pm são as
colunas com mas dados faltantes, vamos quantificalos agora, somando e colocando em formato de
porcentagem"""

totfaltante= oversampleando.isnull().sum().sort_values(ascending= False)

porcentfaltante= (oversampleando.isnull().sum()/arquivo.isnull().count()).sort_values(ascending= False)

perda= pd.concat([totfaltante,porcentfaltante], axis=1, keys=['Total', 'Percent'])

perda.head(4)

print(perda)

#### Foi visto então que as categorias faltantes possuem menos de 50% dos dados então os dados faltants
#### Não seram rejeitados por isso

""" Agora vamos ter 3 trabalhos a fazer aqui
1- Converer colunas em números
2- Converter dados Faltantes em números
3- Tratar Outliers

Começando """

#Mostrando as colunas que são objetos onde eu vou ter que transformar em números
print(oversampleando.select_dtypes(include=['object']).columns)


#Agora eu vou preencher cada NA (valor nullo) com 0

oversampleando['Date'] = oversampleando['Date'].fillna(oversampleando['Date'].mode()[0])

oversampleando['Location'] = oversampleando['Location'].fillna(oversampleando['Location'].mode()[0])

oversampleando['WindGustDir'] = oversampleando['WindGustDir'].fillna(oversampleando['WindGustDir'].mode()[0])

oversampleando['WindDir9am'] = oversampleando['WindDir9am'].fillna(oversampleando['WindDir9am'].mode()[0])

oversampleando['WindDir3pm'] = oversampleando['WindDir3pm'].fillna(oversampleando['WindDir3pm'].mode()[0])

""" LABEL ENCODER 
é a prática de trasnformar uma coluna de string em número para que o modelo consiga executar
"""

# Convert categorical features to continuous features with Label Encoding
from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversampleando.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversampleando[col] = lencoders[col].fit_transform(oversampleando[col])
    
#TRATANDO WARNINGS

import warnings
warnings.filterwarnings("ignore")
# Multiple Imputation by Chained Equations
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
MiceImputed = oversampleando.copy(deep=True) 
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversampleando)

#nesta momento o Dataframe não têm mais valores NaN

#detectando Outiliers
#Primeiro encontrando o primeiro Quartil depois o terceiro e aí então faz o intervalo

Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
print("INTERVALO \n",IQR)

#Removendo Outliers

MiceImputed = MiceImputed[~((MiceImputed < (Q1 - 1.5 * IQR)) |(MiceImputed > (Q3 + 1.5 * IQR))).any(axis=1)]

#printando a remoção dos outliers
print(MiceImputed.shape)

#Agora eu necessito fazer a correlação dos Dados 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
corr = MiceImputed.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(250, 25, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .9})

""" Feature Selection é o processo de seleção de atributos relevantes em um conjunto de dados. 
É uma técnica de pré-processamento de dados que visa melhorar a eficiência, a precisão e a 
interpretabilidade de modelos de aprendizado de máquina, selecionando apenas as características 
que são mais informativas e relevantes para a tarefa específica."""

# Normalizando os dados
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(MiceImputed)
modified_data = pd.DataFrame(r_scaler.transform(MiceImputed), index=MiceImputed.index, columns=MiceImputed.columns)

# Feature Importance using Filter Method (Chi-Square)
from sklearn.feature_selection import SelectKBest, chi2
X = modified_data.loc[:,modified_data.columns!='RainTomorrow']
y = modified_data[['RainTomorrow']]
selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
X_new = selector.transform(X)
print(X.columns[selector.get_support(indices=True)])

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as rf

X = MiceImputed.drop('RainTomorrow', axis=1)
y = MiceImputed['RainTomorrow']
selector = SelectFromModel(rf(n_estimators=100, random_state=0))
selector.fit(X, y)
support = selector.get_support()
features = X.loc[:,support].columns.tolist()
print(features)
print(rf(n_estimators=100, random_state=0).fit(X,y).feature_importances_)


####### Predizendo ####################################################################################################

features = MiceImputed[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
target = MiceImputed['RainTomorrow']

# Split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=12345)

# Normalize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def plot_roc_cur(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
import time
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, plot_confusion_matrix, roc_curve, classification_report
def run_model(model, X_train, y_train, X_test, y_test, verbose=True):
    t0=time.time()
    if verbose == False:
        model.fit(X_train,y_train, verbose=0)
    else:
        model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred) 
    coh_kap = cohen_kappa_score(y_test, y_pred)
    time_taken = time.time()-t0
    print("Accuracy = {}".format(accuracy))
    print("ROC Area under Curve = {}".format(roc_auc))
    print("Cohen's Kappa = {}".format(coh_kap))
    print("Time taken = {}".format(time_taken))
    print(classification_report(y_test,y_pred,digits=5))
    
    probs = model.predict_proba(X_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs) 
    plot_roc_cur(fper, tper)
    
    plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues, normalize = 'all')
    
    return model, accuracy, roc_auc, coh_kap, time_taken

run_model(model, X_train, y_train, X_test, y_test)
plot_confusion_matrix(args, kwargs)
















