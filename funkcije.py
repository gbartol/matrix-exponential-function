import numpy as np
import math
from scipy import linalg
import sys

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
#def expm_pade(A, p = 6, q = 6):
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

    
def expm_rk4(A, v, t, steps):
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

    while(j>1):
        matrica_A = matrica_A/2
        j-=1

    Rqq=Pade_aproksimacija(matrica_A,q,q)

    while(j<pade_vrijednosti[red][stupac][1]):
        Rqq = Rqq.dot(Rqq)
        j += 1

    return Rqq

def Chebysheva_Aproksimacija(matrica_A,q):
    
    #Provjera je li matrica hermitska i jesu li joj sve svojstvene vrijednosti <0
    if linalg.ishermitian(matrica_A) is False:
        print("Matrica nije hermitska. Chebysheva aproksimacija ne radi za takve matrice\n")
        sys.exit(1)
    eigenvalues = np.linalg.eigvalsh(matrica_A)

    for eig in eigenvalues:
        if eig>=0:
            print("Matrica ima svojstvene vrijednosti koje su >=0. Chebysheva aproksimacija ne radi za takve matrice\n")
            sys.exit(1)
    cody_coeffs = {
    1: {
        'a': [1.0669,-1.1535],
        'b': [1.0,1.7275]
    },
    2: {
        'a': [9.92641, -1.88332, 4.21096],
        'b': [1.0, 6.69295, 5.72258]
    },
    3: {
        'a': [1.000799, -2.236578, 1.249962, -9.981009],
        'b': [1.0, 7.982923, 2.204115, 1.24858]
    },
    7: {
        'a': [
            1.000000109,                 # iz 0 (10^0)
            -2.613998104e-1,             # iz 1 (10^-1)
            2.754893180e-2,              # iz 2 (10^-2)
            -1.467589943e-3,             # iz 3 (10^-3)
            4.060544787e-5,              # iz 4 (10^-5)
            -5.370676308e-7,             # iz 5 (10^-7)
            2.653910891e-9,              # iz 6 (10^-9)
            -2.118933743e-12             # iz 7 (10^-12)
        ],
        'b': [
            1.0,                         # iz 0 (10^0)
            7.386066424e-1,              # iz 1 (10^-1)
            2.660947238e-1,              # iz 2 (10^-1)
            6.221006831e-2,              # iz 3 (10^-2)
            1.022960372e-2,              # iz 4 (10^-2)
            1.487848134e-3,              # iz 5 (10^-3)
            8.088769796e-5,              # iz 6 (10^-5)
            1.948334848e-5               # iz 7 (10^-5)
        ]
    },
    14: {
        'a': [
            9.99999999998168e-1,     # iz 0 ( 10^-1 )[cite: 3]
            -2.7495604296300043e-1,  # iz 1 ( -01 )[cite: 3]
            3.4346984175671475e-2,   # iz 2 ( -02 )[cite: 3]
            -2.5674439819028618e-3,  # iz 3 ( -03 )[cite: 3]
            1.2734070715233181e-4,   # iz 4 ( -04 )[cite: 3]
            -4.3932808492511236e-6,  # iz 5 ( -06 )[cite: 3]
            1.0753202054485227e-7,   # iz 6 ( -07 )[cite: 3]
            -1.8710255961089453e-9,  # iz 7 ( -09 )[cite: 3]
            2.2849515765300155e-11,  # iz 8 ( -11 )[cite: 3]
            -1.9038640942835345e-13, # iz 9 ( -13 )[cite: 3]
            1.0310151365350495e-15,  # iz 10 ( -15 )[cite: 3]
            -3.3490280333667533e-18, # iz 11 ( -18 )[cite: 3]
            5.6743825539523501e-21,  # iz 12 ( -21 )[cite: 3]
            -3.7794523874503295e-24, # iz 13 ( -24 )[cite: 3]
            4.1609826642376613e-28   # iz 14 ( -28 )[cite: 3]
        ],
        'b': [
            1.0,                     # iz 0 ( 00 )[cite: 3]
            7.250439570348866e-1,    # iz 1 ( -01 )[cite: 3]
            2.5939094125018012e-1,   # iz 2 ( -01 )[cite: 3]
            6.0968185127283595e-2,   # iz 3 ( -02 )[cite: 3]
            1.0574049161691156e-2,   # iz 4 ( -02 )[cite: 3]
            1.4405527154587316e-3,   # iz 5 ( -03 )[cite: 3]
            1.6019211440612190e-4,   # iz 6 ( -04 )[cite: 3]
            1.4908234724800242e-5,   # iz 7 ( -05 )[cite: 3]
            1.1820291576355772e-6,   # iz 8 ( -06 )[cite: 3]
            8.0163997982357503e-8,   # iz 9 ( -08 )[cite: 3]
            4.8361800878648281e-9,   # iz 10 ( -09 )[cite: 3]
            2.2472030400428529e-10,  # iz 11 ( -10 )[cite: 3]
            1.2922802705792779e-11,  # iz 12 ( -11 )[cite: 3]
            1.8921838854022449e-13,  # iz 13 ( -13 )[cite: 3]
            2.2710680218891295e-14   # iz 14 ( -14 )[cite: 3]
        ]
    }
    }

    if q not in cody_coeffs:
        print(f"Red q={q} nije definiran u tablici. U tablici su samo 1,2,3,7,14")
        sys.exit(1)

    a = cody_coeffs[q]['a']
    b = cody_coeffs[q]['b']

    P_A = np.zeros_like(matrica_A, dtype=float)
    Q_A = np.zeros_like(matrica_A, dtype=float)
    for j in range(len(a)):
        P_A += a[j] * np.linalg.matrix_power(matrica_A, j)
        Q_A += b[j] * np.linalg.matrix_power(matrica_A, j)
        
        
    Cqq = linalg.solve(Q_A, P_A)

    exp_A = np.linalg.inv(Cqq)
    return exp_A

def Lagrange(matrica_A,t=1):

    eigenvalues = np.linalg.eigvals(matrica_A)
    d=matrica_A.shape[0]
    n = len(eigenvalues)
    exp_A = np.zeros((d, d))
    I = np.eye(d)
    for j in range(n):
        koef = math.e**(eigenvalues[j]*t)
        temp_A = np.eye(d)
        for k in range(n):
            if k!=j:
                temp_A = temp_A @ ((matrica_A-I*eigenvalues[k])/(eigenvalues[j]-eigenvalues[k]))
        exp_A = exp_A + koef*temp_A

    return exp_A

def Pomocna_rekurzija(niz,t,pocetak,kraj):
    if(pocetak+1 == kraj):
        return (math.e**(niz[pocetak]*t)-math.e**(niz[kraj]*t))/(niz[pocetak]-niz[kraj])
    else:
        return (Pomocna_rekurzija(niz,t,pocetak,kraj-1)-Pomocna_rekurzija(niz,t,pocetak+1,kraj))/(niz[pocetak]-niz[kraj])
    
def Newton(matrica_A,t=1):
    n = matrica_A.shape[0]
    eigenvalues = np.linalg.eigvals(matrica_A)
    eigenvalues = np.insert(eigenvalues,0,0)
    I = np.eye(n)

    exp_A = math.e**(eigenvalues[1]*t)*I

    temp_A = np.zeros(n)
    for j in range(2,len(eigenvalues)):

        rek = Pomocna_rekurzija(eigenvalues,t,1,j)

        fakt=I

        for k in range(1,j):
            fakt = fakt @ (matrica_A-eigenvalues[k]*I)
        
        temp_A = temp_A + rek*fakt
    
    exp_A = exp_A + temp_A
    return exp_A