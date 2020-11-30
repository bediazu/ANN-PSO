import numpy as np
import pandas as pd

def metrica(y_esperado, y_obtenido):
    vp = 0
    fp = 0
    fn = 0
    vn = 0
    for i in range(len(y_esperado)):
        if y_esperado[i] and y_obtenido[i] == 1:
            vp += 1
        if y_esperado[i] and y_obtenido[i] == -1:
            vn += 1
        if y_esperado[i] == 1 and y_obtenido[i] == -1:
            fn += 1
        if y_esperado[i] == -1 and y_obtenido[i] == 1:
            fp += 1
    accuracy = vp/(vp+fp)
    recall = vp/(vp+fn)
    f_score = 2*(accuracy*recall/(accuracy+recall))


    accuracy_2 = vn/(vn+fp)
    recall_2 = vn/(vn+fn)
    f_score_2 = 2 * (accuracy_2*recall_2)/(accuracy_2+recall_2)
    
    print("Accuracy = "+ str(accuracy))
    print("F_score = " + str(f_score))
    print("F_score Ataque = ", str(f_score_2))
        
    return accuracy, f_score


def Activation(x_n, w_j):
    z = np.dot(w_j, x_n.T)
    for number in z:
      number = np.tanh(number)
    return z

#LOAD param_config
df_config = pd.read_csv('config.csv',header=None)
print('Cargando parametros de configuración...\n',df_config)

L = int(df_config[1][0]) #L: Numero de nodo ocultos
#P = int(df_config[1][1]) #P: Maximo de iteraciones 
#M = int(df_config[1][2]) #M : Numero de particulas
#C = int(df_config[1][3]) #C : Penalidad Pseudo-Inversa
  
# Load test file
df_test = pd.read_csv('test', sep=',' ,header=None)
df_test.replace(np.nan,0.1,inplace=True)

test = df_test.to_numpy()

# Load test_input
xv = test[1:,1:-1]

[N, D] = xv.shape
# Generate bias unit
temp_bias = np.ones((N, 1))

# Add bias unit
Xv = np.hstack((xv, temp_bias))

# Load test_label
yv = test[1:,-1]

#cargar pesos entrenados
container = np.load("pesos.npz")  

weight_data = [container[key] for key in container]
w1 = weight_data[0]
w2 = weight_data[1]
MSE = weight_data[2]


w1 = w1.reshape((L,D+1))
H = Activation(Xv, w1)
zv = np.dot(w2,H)

#se transforma el valor dependiendo de si es mayor o menor a 0 en prediccion
for number in range(len(zv)):
    if zv[number] < 0: 
        zv[number] = -1
    else:
        zv[number] = 1

#Utilizar nuestras propias metricas 
accuracy, f_score = metrica(yv, zv)

