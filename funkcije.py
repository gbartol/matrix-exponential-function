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

def Pade_aproksimacija(matrica_A,p,q):
    #Pade dekompozicija algoritam
    dimenzija = len(matrica_A)
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
    # SKALIRANJE
    # e^A = (e^(A/m))^m
    # m je najmanji pow 2 t.d. ||A||/m <= 1
    # cancelation error i loša uvjetovanost nazivnika Dpq
    norma_A = linalg.norm(matrica_A, ord=np.inf)
    m = 0
    if norma_A > 0.5:
        m = int(np.ceil(np.log2(norma_A/0.5)))
        matrica_A = matrica_A /(2.0 ** m)


    #Pade dekompozicija algoritam
    dimenzija = len(matrica_A)
    Npq = np.zeros((dimenzija,dimenzija))
    Dpq = np.zeros((dimenzija,dimenzija))
    potencija_A = np.array(matrica_A, dtype=float)  

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

    # Vračanje originala
    for _ in range(m):
        Rpq = Rpq.dot(Rpq)

    return Rpq
def expm_pade(A, p = 6, q = 6):
    #TO DO 

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

    
def expmv_rk4(A, v, t, steps):
    h = t / steps
    y = np.asarray(v).copy()

    for _ in range(steps):
        k1 = h * (A @ y)
        k2 = h * (A @ (y + k1 / 2))
        k3 = h * (A @ (y + k2 / 2))
        k4 = h * (A @ (y + k3))
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return y

def expm_arnoldi(A,v,t = 1.0, m=30):
    V, H, beta, breakdown = arnoldi_iteration(A,v,m)
    k = H.shape[1]
    H_k = H[:k, :k]
    exp_H = Pade_simetricni(t*H_k, 6) # TO DO, iskoristi Vidovo remek dijelo
    exp_H = linalg.expm(t*H_k)

    #w = beta * V_k * exp(t * H_k) * e_1
    y = exp_H[:, 0]
    w = beta * (V[:, :k] @ y)

    if breakdown:
        err_est = 0.0
    else:
        h_next = H[k, k-1]
        err_est = beta * h_next * abs(exp_H[k-1, 0])
    
    return w, err_est

def Skalirani_Pade(matrica_A):
    norma_A = math.sqrt(max(linalg.eigvals(np.transpose(matrica_A) .dot(matrica_A))).real)
    pade_vrijednosti = np.array([
        [[1, 0], [1, 0], [2, 0], [3, 0], [3, 0]],
        [[1, 0], [2, 0], [3, 0], [4, 0], [4, 0]],
        [[2, 1], [3, 1], [4, 1], [5, 1], [6, 1]],
        [[2, 5], [3, 5], [4, 5], [5, 5], [6, 5]],
        [[2, 8], [3, 8], [4, 8], [5, 8], [6, 8]],
        [[2, 11], [3, 11], [4, 11], [5, 11], [6, 11]]
    ])

    norma_A_om = int(round(math.log10(norma_A),0))

    red = norma_A_om+2
    stupac = 2

    q = pade_vrijednosti[red][stupac][0]
    j = pade_vrijednosti[red][stupac][1]
    m=1
    while(j>1):
        matrica_A = matrica_A/2
        j-=1
        m *=2

    Rqq=Pade_aproksimacija(matrica_A,q,q)

    while(j<pade_vrijednosti[red][stupac][1]):
        Rqq = Rqq.dot(Rqq)
        j += 1

    return Rqq
