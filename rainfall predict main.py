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
plt.show()


""" É observado que existe uma descrepância em relação aos dados da classe pois os do Tipo 0 são muitos e 1 poucos
concluímos que o dataset esta desbalanceado, e precisamos balancea-ló""" 

# Utilizando a biblioteca Sklearn e pegando o metodo reseample para equalizar a classe desbalanceada

from sklearn.utils import resample


zeros= arquivo[arquivo.RainTomorrow == 0]
ums= arquivo[arquivo.RainTomorrow == 1]

oversampleando =  resample(ums, replace=True, n_samples=len(zeros), random_state=123)

feito= pd.concat([zeros, oversampleando])

fig = plt.figure(figsize = (8,5))

feito.RainTomorrow.value_counts(normalize = True).plot(kind='bar', color= ['skyblue','navy'], alpha = 0.9, rot=0)
plt.title('RainTomorrow Indicator No(0) and Yes(1) after Oversampling (Balanced Dataset)')
plt.show()


#primeiramente separar os dados das classes desbalanceadas


















