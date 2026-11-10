import numpy as np
import matplotlib.pyplot as plt
from funkcije import expm_rk4


A = np.array([
    [-1, 1],
    [0, -2]
    ])
v = np.transpose([0, 1])
t = 1.0
s = [i for i in range(1, 32)]

y_nums = []
for i in s:
    y_nums.append(expm_rk4(A, v, t, steps=i))
y_ref = np.array([np.exp(-t) - np.exp(-2 * t), np.exp(-2 * t)])

rel_err = [np.linalg.norm(y_num - y_ref) / np.linalg.norm(y_ref) for y_num in y_nums]

plt.plot(
        s, rel_err
        )
