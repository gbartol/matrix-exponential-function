import numpy as np
import time
from scipy import linalg

from funkcije import (
    expm_taylor,
    Skalirani_taylor,
    Pade_aproksimacija,
    Lagrange,
    Hessenberg,
    Tri_Sys_Eig
)

def test_osnovna_tocnost(funckija):
    print("=" * 60)
    print(f"TEST: Osnovna točnost na gustoj matrici za funckiju {funckija.__name__}")
    print("=" * 60)
    
    np.random.seed(42)
    n = 50
    m = 15
    t = 1.0
    
    # Nasumična matrica skalirana da ima umjerenu normu
    A = np.random.randn(n, n) / np.sqrt(n)
    v = np.random.randn(n)
    
    # 1. aproksimacija dane funkcije
    t0 = time.perf_counter()
    w_test= funckija(A)
    t_test = time.perf_counter() - t0
    
    # 2. Referentno rješenje preko punog SciPy expm
    t0 = time.perf_counter()
    w_ref = linalg.expm(t * A) 
    t_ref = time.perf_counter() - t0

    err_test = linalg.norm(w_test - w_ref)
    rel_err_test = linalg.norm(w_test - w_ref) / linalg.norm(w_ref)

    
    print(f"Dimenzija matrice: {n}x{n}, potprostor m = {m}, t = {t}")
    print(f"Stvarna pogreška funckije {funckija.__name__}:     {err_test:.2e}")
    print(f"Relativna pogreška funckije {funckija.__name__}:  {rel_err_test:.2e}")
    print(f"Vrijeme funckije f{funckija.__name__}:             {t_test * 1e3:.3f} ms")
    print(f"Vrijeme SciPy expm (puna):   {t_ref * 1e3:.3f} ms")

if __name__ == "__main__":
    test_osnovna_tocnost(expm_taylor)
    test_osnovna_tocnost(Skalirani_taylor)
    test_osnovna_tocnost(Pade_aproksimacija)
    test_osnovna_tocnost(Lagrange)
    test_osnovna_tocnost(Hessenberg)
    test_osnovna_tocnost(Tri_Sys_Eig)

