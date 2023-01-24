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