#Trabalho elaborado por: Renan Pacheco e Taine Freitas
#O trabalho consegue fazer ate x5

from __future__ import division  
import numpy as np  
from numpy import linalg 
import matplotlib.pyplot as plt

def jacobi(A,b,x0,tol,N):  
    #preliminares
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double') 
    vet_x1=[x0[0,0]]
    vet_x2=[x0[1,0]]


    n=np.shape(A)[0]  
    x = np.zeros(n)  
    it = 0  
    #iteracoes  
    while (it < N):  
        it = it+1  
        #iteracao de Jacobi  
        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,n))):  
                x[i] -= A[i,j]*x0[j]  
            x[i] /= A[i,i]
            vet_x1.append(x[0])
            vet_x2.append(x[1])
        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            reta1_x = [0,0]
            reta2 = [0,0]
            #Pontos para o traco das retas
            #Reta 1
            reta1_x[0] = (b[0] - (-5*A[0, 1]))/A[0,0]
            reta1_x[1]= (b[1] - (5* A[0, 1]))/A[0,0]
            reta_y = [-5,5]
            #Reta 2
            reta2[0] = (b[0]*A[1,0])/A[1,1]
            reta2[1] =  (b[1]*A[1,0])/A[1,1]

            plt.plot(reta1_x,reta_y, reta2, reta_y)
            plt.plot(vet_x1, vet_x2, 'ro')            
            plt.title("Resultados Jacobi")
            plt.show()
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  
#   raise NameError('num. max. de iteracoes excedido.')

def gauss_seidel(A,b,x0,tol,N):  
    #preliminares  
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  
    vet_x1=[x0[0,0]]
    vet_x2=[x0[1,0]]

    n=np.shape(A)[0]  
    x = np.copy(x0)  
    it = 0  
    #iteracoes  
    while (it < N):  
        it = it+1  
        #iteracao de Jacobi  
        for i in np.arange(n):  
            x[i] = b[i]  
            for j in np.concatenate((np.arange(0,i),np.arange(i+1,n))):  
                x[i] -= A[i,j]*x[j]  
            x[i] /= A[i,i]
            vet_x1.append(x[0])
            vet_x2.append(x[1])

        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol): 
            reta1_x = [0,0]
            reta2 = [0,0]
            #Pontos para o traco das retas
            #Reta 1
            reta1_x[0] = (b[0] - (-5*A[0, 1]))/A[0,0]
            reta1_x[1]= (b[1] - (5* A[0, 1]))/A[0,0]
            reta_y = [-5,5]
            #Reta 2
            reta2[0] = (b[0]*A[1,0])/A[1,1]
            reta2[1] =  (b[1]*A[1,0])/A[1,1]

            plt.plot(reta1_x,reta_y, reta2, reta_y)

            plt.plot(vet_x1, vet_x2, 'ro')
            plt.title("Resultados Seidel")
            plt.show()
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  
 #   raise NameError('num. max. de iteracoes excedido.')
A = np.array([[1,1],
            [1, -3]],
            dtype='double')


b = np.array([[3],[-3]])
tol = 0.001
N = 25
x0 = np.array([[0], [0]])

x_jacobi = jacobi(A, b, x0, tol, N)
x_gauss = gauss_seidel(A, b, x0, tol, N)
