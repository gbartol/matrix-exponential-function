<<<<<<< Updated upstream
# Arnoldi.py
import numpy as np
import time
from scipy import linalg
from scipy.sparse import diags

from funkcije import (
    arnoldi_iteration,
    expm_arnoldi,
    expmv_rk4,
    Skalirani_Pade
)

def test_osnovna_tocnost():
    print("=" * 60)
    print("TEST 1: Osnovna točnost na gustoj matrici")
    print("=" * 60)
    
    np.random.seed(42)
    n = 50
    m = 15
    t = 1.0
    
    # Nasumična matrica skalirana da ima umjerenu normu
    A = np.random.randn(n, n) / np.sqrt(n)
    v = np.random.randn(n)
    
    # 1. Arnoldijeva aproksimacija (Metoda 20)
    t0 = time.perf_counter()
    w_arnoldi, err_est = expm_arnoldi(A, v, t=t, m=m)
    t_arnoldi = time.perf_counter() - t0
    
    # 2. Referentno rješenje preko punog SciPy expm
    t0 = time.perf_counter()
    w_ref = linalg.expm(t * A) @ v
    t_ref = time.perf_counter() - t0
    
    # 3. Usporedba s Runge-Kutta 4 (iz funkcije.py)
    w_rk4 = expmv_rk4(A, v, t, steps=200)
    
    rel_err_arnoldi = linalg.norm(w_arnoldi - w_ref) / linalg.norm(w_ref)
    rel_err_rk4 = linalg.norm(w_rk4 - w_ref) / linalg.norm(w_ref)
    
    print(f"Dimenzija matrice: {n}x{n}, potprostor m = {m}, t = {t}")
    print(f"Relativna pogreška Arnoldi:  {rel_err_arnoldi:.2e}")
    print(f"Saadova procjena pogreške:   {err_est:.2e}")
    print(f"Relativna pogreška RK4:      {rel_err_rk4:.2e}")
    print(f"Vrijeme Arnoldi:             {t_arnoldi * 1e3:.3f} ms")
    print(f"Vrijeme SciPy expm (puna):   {t_ref * 1e3:.3f} ms")


def test_konvergencija_po_m():
    print("\n" + "=" * 60)
    print("TEST 2: Konvergencija pogreške u ovisnosti o dimenziji m")
    print("=" * 60)
    
    np.random.seed(0)
    n = 60
    t = 1.0
    A = np.random.randn(n, n) / np.sqrt(n)
    v = np.random.randn(n)
    
    w_ref = linalg.expm(t * A) @ v
    norm_w_ref = linalg.norm(w_ref)
    
    print(f"{'m':>4} | {'Stvarna rel. greška':>20} | {'Saadova procjena':>20}")
    print("-" * 50)
    
    for m in range(2, 22, 2):
        w_m, err_est = expm_arnoldi(A, v, t=t, m=m)
        stvarna_greska = linalg.norm(w_m - w_ref) / norm_w_ref
        print(f"{m:4d} | {stvarna_greska:20.2e} | {err_est:20.2e}")


def test_lucky_breakdown():
    print("\n" + "=" * 60)
    print("TEST 3: Provjera 'Lucky Breakdown' svojstva")
    print("=" * 60)
    
    # Ako je v svojstveni vektor matrice A (A @ v = lambda * v),
    # dimenzija Krilovljevog potprostora je točno 1.
    diag_vals = [1.0, 2.5, -3.0, 4.0, 5.0]
    A = np.diag(diag_vals)
    v = np.array([0.0, 1.0, 0.0, 0.0, 0.0])  # Svojstveni vektor za lambda = 2.5
    t = 0.5
    
    w_arnoldi, err_est = expm_arnoldi(A, v, t=t, m=4)
    w_analiticko = np.exp(2.5 * t) * v
    
    rel_err = linalg.norm(w_arnoldi - w_analiticko) / linalg.norm(w_analiticko)
    print(f"Zadana dimenzija potprostora: m = 4")
    print(f"Stvarna relativna pogreška:   {rel_err:.2e}")
    print(f"Saadova procijenjena greška:  {err_est:.2e}")
    print("Metoda se automatski zaustavila na k = 1 s točnim rješenjem.")


def test_velika_rijetka_matrica():
    print("\n" + "=" * 60)
    print("TEST 4: Skalabilnost na većoj tridijagonalnoj matrici (1D Laplace)")
    print("=" * 60)
    
    # Diskretizacija difuzijskog operatora u 1D: A = tridiag(1, -2, 1)
    n = 2000
    m = 25
    t = 0.1
    
    glavna_dijagonala = -2.0 * np.ones(n)
    sporedne_dijagonale = np.ones(n - 1)
    L = diags([sporedne_dijagonale, glavna_dijagonala, sporedne_dijagonale], 
              [-1, 0, 1], format='csr')
    
    v = np.ones(n)
    
    t0 = time.perf_counter()
    w_arnoldi, err_est = expm_arnoldi(L, v, t=t, m=m)
    t_arnoldi = time.perf_counter() - t0
    
    print(f"Dimenzija matrice: {n}x{n}")
    print(f"Dimenzija Krilovljevog potprostora: m = {m}")
    print(f"Vrijeme izvođenja: {t_arnoldi * 1e3:.2f} ms")
    print(f"Saadova procijenjena pogreška: {err_est:.2e}")
    print(f"Norma izračunatog vektora ||w||: {linalg.norm(w_arnoldi):.4f}")
    print("Napomena: Računanje pune matrice exp(t*L) za n=2000 uništilo bi rijetkost i zahtijevalo ~32 MB memorije.")


if __name__ == "__main__":
    test_osnovna_tocnost()
    test_konvergencija_po_m()
    test_lucky_breakdown()
    test_velika_rijetka_matrica()
=======
import numpy as np
import math
from scipy import linalg
>>>>>>> Stashed changes
