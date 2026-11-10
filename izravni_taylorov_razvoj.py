import numpy as np
import math
from scipy import linalg

def Izravni_taylorov_razvoj(matrica_A,broj_iteracija):
    exp_A = np.eye(2, dtype=float)   
    potencija_A = np.array(matrica_A, dtype=float) 
    
    for i in range(1,broj_iteracija+1):
        fakt = float(math.factorial(i))
        exp_A += potencija_A/fakt
        potencija_A = potencija_A.dot(A)
    return exp_A
def Pade_aproksimacija(matrica_A,p,q):
    #Pade dekompozicija algoritam
    dimenzija = len(A)
    Npq = np.zeros((dimenzija,dimenzija))
    Dpq = np.zeros((dimenzija,dimenzija))


    for j in range(0,p+1):
        p_q_j = float(math.factorial(p+q-j))
        p_ = float(math.factorial(p))
        p_q = float(math.factorial(p+q))
        j_ = float(math.factorial(j))
        p_j = float(math.factorial(p-j))
        potencija_A = np.linalg.matrix_power(matrica_A, j)
        Npq += potencija_A*((p_q_j*p_)/(p_q*j_*p_j))

    for j in range(0,q+1):
        p_q_j = float(math.factorial(p+q-j))
        q_ = float(math.factorial(q))
        p_q = float(math.factorial(p+q))
        j_ = float(math.factorial(j))
        q_j = float(math.factorial(q-j))
        potencija_A = np.linalg.matrix_power(-matrica_A, j)
        Dpq += potencija_A*((p_q_j*q_)/(p_q*j_*q_j))

    Rpq = linalg.solve(Dpq, Npq)
    return Rpq

def Pade_simetricni(matrica_A,q):
    #Pade dekompozicija algoritam
    dimenzija = len(matrica_A)
    Npq = np.zeros((dimenzija,dimenzija))
    Dpq = np.zeros((dimenzija,dimenzija))
    potencija_A = np.array(matrica_A, dtype=float)  
    potencija_A_neg = np.array(matrica_A, dtype=float) 
    for j in range(0,q+1):
        q_q_j = float(math.factorial(q+q-j))
        q_ = float(math.factorial(q))
        q_q = float(math.factorial(q+q))
        j_ = float(math.factorial(j))
        q_j = float(math.factorial(q-j))
        potencija_A = np.linalg.matrix_power(matrica_A, j)
        Npq += potencija_A*((q_q_j*q_)/(q_q*j_*q_j))
        potencija_A_neg = np.linalg.matrix_power(-matrica_A, j)
        Dpq += potencija_A_neg*((q_q_j*q_)/(q_q*j_*q_j))

    Rpq = linalg.solve(Dpq, Npq)
    return Rpq
if __name__ == "__main__":
    #Pade vrijednosti za najbolju procjenu eksponencijalne matrice
    #Za kasnije
    pade_vrijednosti = np.array([
        [[1, 0], [1, 0], [2, 0], [3, 0], [3, 0]],
        [[1, 0], [2, 0], [3, 0], [4, 0], [4, 0]],
        [[2, 1], [3, 1], [4, 1], [5, 1], [6, 1]],
        [[2, 5], [3, 5], [4, 5], [5, 5], [6, 5]],
        [[2, 8], [3, 8], [4, 8], [5, 8], [6, 8]],
        [[2, 11], [3, 11], [4, 11], [5, 11], [6, 11]]
    ])

    A = [[-49,24],
        [-64,31]]
    A = np.array(A, dtype=float)
    A_transponirano = np.transpose(A)

    exp_A = np.eye(2, dtype=float)      
    potencija_A = np.array(A, dtype=float)  
    broj_iteracija=int(input("Koliki broj iteracija za Taylorov red\n"))

    exp_A = Izravni_taylorov_razvoj(A,broj_iteracija)

    print("\nMatrica exp(A) dobivena Taylorovim redom")
    for i in range(len(exp_A)):
        print(exp_A[i])

    print("\nSvojstvene vrijednosti matrice A")
    print(linalg.eigvals(A))    
    print("\nNorma matrice A")
    print(math.sqrt(max(linalg.eigvals(A_transponirano.dot(A))).real))

    #p i q takvi jer je norma matrice A ~10^2 a želimo najmanju grešku
    p = 6
    q = 8

    Rpq = Pade_aproksimacija(A,p,q)

    print("\nMatrica exp(A) dobivena Pade dekompozocijom")
    for i in range(len(Rpq)):
        print(Rpq[i])
    print("\nRazlika Taylorovog reda i Pade dekompozicije")
    for i in range(len(Rpq)):
        print("[",end=" ")
        for j in range(len(Rpq[i])):
            print(exp_A[i][j]-Rpq[i][j],end=" ")
        print("]")