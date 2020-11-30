import numpy as np
import pandas as pd
import random


class PSO:
  def __init__(self, maxIter, numPart, numHidden, D, Xe, ye, C):
    self.maxIter = maxIter
    self.np = numPart
    self.nh = numHidden
    self.D = D
    self.xe = Xe
    self.ye = ye
    self.C = C

    self.w1 = None
    self.w2 = None

    self.ini_swarm(numPart, numHidden, D)
    print('Instancia del PSO generada con exito...')

  def ini_swarm(self, numPart, numHidde, D):
    # Se inicializa una matriz con las dimensiones nh * D
    dim = self.nh * D
    X = np.zeros( (self.np, dim), dtype=float)

    #Se asignan pesos aleatorios para cada particula
    for i in range(self.np):
      wh = self.rand_w(self.nh, D)
      a = np.reshape(wh, (1, dim))
      X[i] = a

    # Se almacena la matriz de particulas generadas
    self.X = X

  def Activation(self, x_n, w_j):
    z = np.dot(w_j, x_n.T)
    for number in z:
      number = np.tanh(number)
    return z

  
  def fitness(self):
    w2 = np.zeros( (self.np, self.nh), dtype=float)
    MSE = np.zeros(self.np, dtype=float)

    for i in range(self.np):
      p = self.X[i]
      w1 = np.reshape(p, (self.nh, self.D))
      H = self.Activation(self.xe, w1)
      w2[i] = self.mlp_pinv(H)
      ze = np.dot(w2[i], H)
      MSE[i] = np.square(np.subtract(self.ye,ze)).mean()
    
    return MSE, w2

  
  def mlp_pinv(self, H):
    L, N = H.shape
    yh = np.dot(self.ye.T, H.T)
    hh = np.dot(H, H.T)
    hh = hh + (np.eye(hh.shape[0])/self.C)
    w2 = np.dot(yh.T, np.linalg.pinv(hh))

    return w2

  def rand_w(self, nextNodes, currentNodes):
    w = np.random.random((nextNodes, currentNodes))
    x = nextNodes + currentNodes
    r = np.sqrt(6/x)
    w = w * 2 * r - r

    return w
  
  def upd_particle(self, X, pBest, pFitness, gBest, gFitness, New_pFitness, newBeta, wBest):

    for i in range(self.np):
      if (New_pFitness[i] < pFitness[i]):
        pFitness[i] = New_pFitness[i]
        pBest[i][:] = X[i,:]

    New_gFitness = min(pFitness)
    idx = np.argmin(pFitness)
    if (New_gFitness < gFitness):
      gFitness = New_gFitness
      gBest = pBest[idx][:]
      wBest = newBeta[idx][:]

    return pBest, pFitness, gBest, gFitness, wBest

  def run_PSO(self):
    idx = 0

    #Config Swarm
    alpha = np.ones(self.maxIter)
    for i in range(self.maxIter):
      alpha[i] = 0.95-((0.95-0.2)/self.maxIter) * i
    pBest = np.zeros((self.np, self.D*self.nh))
    pFitness = np.ones(self.np)*(100000)
    pBest = np.zeros((self.np,self.D*self.nh))
    gBest = np.zeros(self.D*self.nh)
    gFitness = (100000)
    wBest = np.zeros(self.nh)
    v = np.zeros((self.np,self.D*self.nh))
    Z1 = np.ones((self.np,self.D*self.nh))
    Z2 = np.ones((self.np,self.D*self.nh))
    MSE = np.zeros(self.maxIter)

    for idx in range(self.maxIter):
      print("Iteración Numero: ",idx)
      new_pFitness, newBeta = self.fitness()
      pBest, pFitness, gBest, gFitness, wBest = self.upd_particle(self.X, pBest, pFitness, gBest,gFitness,new_pFitness, newBeta, wBest)

      MSE[idx] = gFitness

      for i in range(self.np):
        for j in range(self.nh*self.D):
          Z1[i,j] = 1.05 * (pBest[i,j]-self.X[i,j]) * random.uniform(0, 1)
          Z2[i,j] = 2.95 * (gBest[j]-self.X[i,j]) * random.uniform(0, 1)
      v = (v + (Z1 + Z2)) * alpha[idx]
      self.X = self.X + v

    df_mse = pd.DataFrame(data=MSE)
    df_mse.to_csv('costos.csv', index=False, header=False)
    return gBest, wBest, MSE