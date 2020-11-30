import pandas as pd
import numpy as np

from Classes.PSO import PSO

#LOAD param_config
df_config = pd.read_csv('config.csv',header=None)
print('Cargando parametros de configuración...\n',df_config)

L = int(df_config[1][0]) #L: Numero de nodo ocultos
P = int(df_config[1][1]) #P: Maximo de iteraciones 
M = int(df_config[1][2]) #M : Numero de particulas
C = int(df_config[1][3]) #C : Penalidad Pseudo-Inversa


# Load train file
df = pd.read_csv('train', sep=',' ,header=None)
df.replace(np.nan,0.1,inplace=True)
train = df.to_numpy()

# Load train_input
xe = train[1:,1:-1]
print('Cargando train_input: ',xe.shape)

[N ,D] = xe.shape
# Generate bias unit
temp_bias = np.ones((N, 1))

# Add bias unit
Xe =  np.hstack((xe, temp_bias))
print('Cargando train_input: ',Xe.shape)
  
# Load train_label
ye = train[1:,-1]
print('Cargando train_label: ',xe.shape)


D += 1

pso = PSO(M, P, L, D, Xe, ye, C)
w1, w2, MSE = pso.run_PSO()


np.savez("pesos", w1 = w1, w2 = w2, MSE = MSE)