import numpy as np
import math
from scipy import linalg

def expm_taylor(A, broj_iteracija = 30):
    n = A.shape[0]
    exp_A = np.eye(n, dtype = float)
    potencija_A = np.array(A, dtype = float)

    for i in range(1, broj_iteracija + 1):
        fakt = float(math.factorial(i))
        exp_A += potencija_A / fakt
        potencija_A = potencija_A.dot(A)

    return exp_A


def arnoldi_iteration(A, v, m):

    """
    Kreira ortonormiranu bazu Krilovljevog potprostora K_m(A, v) i gornju
    Hessenbergovu matricu H koristenjem modificiranog Gram-Schmidtovog postupka.
    
    Parametri:
    ----------
    A : ndarray ili funkcija/operator (n x n)
    v : ndarray duljine n, pocetni vektor
    m : int, zeljena dimenzija potprostora (m < n)
    
    Povratne vrijednosti:
    --------------------
    V : ndarray oblika (n, k + 1), ortonormirani vektori baze (k <= m)
    H : ndarray oblika (k + 1, k), prosirena gornja Hessenbergova matrica
    beta : float, 2-norma pocetnog vektora v
    breakdown : bool, True ako je detektiran invarijantan potprostor
    """


    v = np.asarray(v, dtype = float)
    beta = np.linalg.norm(v)

    if beta == 0.0:
        raise ValueError("v = nulvektor. WHOOOPS")

    n = len(v)

    V = np.zeros((n, m +1), dtype = float)
    H = np.zeros((m+1, m), dtype = float)

    V[:, 0] = v/beta

    breakdown = False

    k = m
    for j in range(m):
        if callable(A):
            w = A(V[:, j])
        else:
            w = A.dot(V[:,j])

        for i in range(j+1):
            H[i,j] = np.dot(V[:,i].conj(), w) # h_ij = <v_i, w>
            w-= H[i,j] * V[:,i] 

        h_next = np.linalg.norm(w)
        H[j+1, j] = h_next

        if h_next< 1e-14:
            breakdown = True
            k = j + 1
            V = V[:, : k+1]
            H = H[:k+1, :k]
            break

        V[:, j+1] = w/h_next

    return V, H, beta, breakdown

def Skalirani_taylor(matrica_A,broj_iteracija): 
    norma_A = math.sqrt(max(linalg.eigvals(np.transpose(matrica_A) .dot(matrica_A))).real)
    m=1
    if(norma_A<=1):
        print("Matrica ima malu normu. Skaliranje nije potrebno. Nastavljam s običnim Taylorovim razvojem\n")

    while(norma_A>1):
        norma_A = norma_A/2
        m *= 2

    matrica_A = matrica_A/m
    exp_A = expm_taylor(matrica_A,broj_iteracija)

    while(m>1):
        exp_A = exp_A.dot(exp_A)
        m=m/2
    
    return exp_A
