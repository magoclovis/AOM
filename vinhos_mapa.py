# biblioteca para trabalhar com mapas auto organizaveis 
from minisom import MiniSom
import pandas as pd

base = pd.read_csv('wines.csv')

# : = todas as linhas
# 1:14 = atributo 1 ao atributo 14 (coluna)
# 14 é o upperbound e não é retornado
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values

from sklearn.preprocessing import MinMaxScaler

normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# som = self organize map
# x,y = altura/largura do mapa/matriz
# input_len = numero de atributos/entradas
# sigma = valor do raio dos neurônios do BMU (best matching unit)
# learning_rate = taxa de aprendizagem
som = MiniSom(x = 8, y = 8, input_len = 13, sigma = 1.0, learning_rate = 0.5, random_seed = 2)

# criando pesos aleatórios para a base de dados
som.random_weights_init(X)

# treinamento
som.train_random(data = X, num_iteration = 100)

som._weights
som._activation_map
q = som.activation_response(X)

from matplotlib.pylab import pcolor, colorbar, plot

# .T = traz a matriz transposta
pcolor(som.distance_map().T)
# MID - mean inter neuron distance = media da distancia entre esses neuronios

# escala
colorbar()

# dizer qual é o neuronio ganhador de cada um dos registros
w = som.winner(X[2])

# mudar o formato de "mapa de calor" para figuras coloridas

# o = circulo
# s = square
# D = diagonal
markers = ['o', 's', 'D']

# rgb
color = ['r', 'g', 'b']

# para comecar no indice 0 ao invés de 1 para executar sem erro
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

# substituir o formato em si
for i, x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
