#Trabalho elaborado por: Renan Pacheco e Taine Freitas
#O trabalho consegue fazer ate x5

from __future__ import division  
import numpy as np  
from numpy import linalg 
import matplotlib.pyplot as plt

vet_x1=[]
vet_x2=[]
vet_x3=[]
vet_x4=[]
vet_x5=[]


def jacobi(A,b,x0,tol,N):  
    #preliminares
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double') 
 
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
            if i == 0:
                vet_x1.append(x[i])
            elif i == 1:
                vet_x2.append(x[i])
            elif i == 2:
                vet_x3.append(x[i])
            elif i == 3:
                vet_x4.append(x[i])
            else:
                vet_x5.append(x[i])
            

        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  
#   raise NameError('num. max. de iteracoes excedido.')

def gauss_seidel(A,b,x0,tol,N):  
    #preliminares  
    A = A.astype('double')  
    b = b.astype('double')  
    x0 = x0.astype('double')  
 
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
            if i == 0:
                vet_x1.append(x[i])
            elif i == 1:
                vet_x2.append(x[i])
            elif i == 2:
                vet_x3.append(x[i])
            elif i == 3:
                vet_x4.append(x[i])
            else:
                vet_x5.append(x[i])
            
        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            #print(x)
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  
 #   raise NameError('num. max. de iteracoes excedido.')
A = np.array([[2,5],
               [3, 1]],
               dtype='double')
#D = np.diag(np.diag(A))
#L = np.tril(A) - D
#U = np.triu(A) - D
b = np.array([-3,2])
tol = 0.001
N = 25
x0 = np.array([5, 4])

x_jacobi = jacobi(A, b, x0, tol, N)
#
# plt.plot[vet_x1]
plt.plot(A)
plt.title("Resultados Jacobi")
plt.show()

x_gauss = gauss_seidel(A, b, x0, tol, N)
#plt.plot(vet_y)
#plt.title("Resultados Seidel")
#plt.show()