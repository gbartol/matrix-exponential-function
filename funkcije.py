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

