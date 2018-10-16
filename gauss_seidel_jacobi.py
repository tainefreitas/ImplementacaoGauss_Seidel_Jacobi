#Trabalho elaborado por: Renan Pacheco e Taine Freitas

from __future__ import division  
import numpy as np  
from numpy import linalg 
import matplotlib.pyplot as plt
vet_x=[]

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
            #print(x[i],A[i,i])
            #plt.plot(x[i])
            vet_x.append(x[i])

        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            print(x)
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
            print(x[i],A[i,i])  
        #tolerancia  
        if (np.linalg.norm(x-x0,np.inf) < tol):  
            print(x)
            return x  
        #prepara nova iteracao  
        x0 = np.copy(x)  
#    raise NameError('num. max. de iteracoes excedido.')
A = np.array([[2,5],
               [3, 1]],
               dtype='double')
#D = np.diag(np.diag(A))
#L = np.tril(A) - D
#U = np.triu(A) - D
b = np.array([-3,2])
tol = 0.00001
N = 25
x0 = np.array([5, 4])

x_jacobi = jacobi(A, b, x0, tol, N)
plt.plot(vet_x)
plt.title("jacobi")
plt.show()
plt.savefig("grafico_jacobi.png")
#Com a funcao exemplo, retorna que deu o nro max de iteracoes, e mesmo alterando, nao 
#modificou o erro
#x_gauss = gauss_seidel(A, b, x0, tol, N)