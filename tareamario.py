# TAREA 1 - SOLUCIONES DE ECUACIONES COMPLETA
# Autor: Emiliano

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# -------------------------
# EJERCICIO 1: Polinomio cuadrático
f1 = lambda x: x**2 -5*x +6
df1 = lambda x: 2*x -5
x1 = np.linspace(0,5,300)
plt.figure(figsize=(6,4))
plt.plot(x1,f1(x1), label="f(x)=x^2-5x+6")
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 1")
plt.grid()
plt.show()
r1b, r2b = optimize.bisect(f1,1,3), optimize.bisect(f1,3,4)
r1n, r2n = optimize.newton(f1,1,fprime=df1), optimize.newton(f1,3,fprime=df1)
r1s, r2s = optimize.newton(f1,1,x1=2), optimize.newton(f1,3,x1=4)
print("EJERCICIO 1")
print("Bisección:",r1b,r2b)
print("Newton:",r1n,r2n)
print("Secante:",r1s,r2s)

# -------------------------
# EJERCICIO 2: Polinomio cúbico
f2 = lambda x: x**3 -6*x**2 +11*x -6
df2 = lambda x: 3*x**2 -12*x +11
x2 = np.linspace(0,4,400)
plt.figure(figsize=(6,4))
plt.plot(x2,f2(x2))
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 2")
plt.grid()
plt.show()
r1b,r2b,r3b = optimize.bisect(f2,0,1.5), optimize.bisect(f2,1.5,2.5), optimize.bisect(f2,2.5,3.5)
r1n,r2n,r3n = optimize.newton(f2,0.5,fprime=df2), optimize.newton(f2,2,fprime=df2), optimize.newton(f2,3.2,fprime=df2)
r1s,r2s,r3s = optimize.newton(f2,0,x1=1), optimize.newton(f2,1.5,x1=2.5), optimize.newton(f2,2.5,x1=3.5)
print("EJERCICIO 2")
print("Bisección:",r1b,r2b,r3b)
print("Newton:",r1n,r2n,r3n)
print("Secante:",r1s,r2s,r3s)

# -------------------------
# EJERCICIO 3: Función dada
f3 = lambda x: x**3 -5*x**2*np.exp(-x) + np.exp(-3*x)
df3 = lambda x: 3*x**2 +5*x*np.exp(-x)*(x-2) -3*np.exp(-3*x)
x3 = np.linspace(0,2,200)
plt.figure(figsize=(6,4))
plt.plot(x3,f3(x3),color="firebrick")
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 3")
plt.grid()
plt.show()
r1b = optimize.bisect(f3,0,1)
r2b = optimize.bisect(f3,1,1.75)
r1n = optimize.newton(f3,0,fprime=df3)
r2n = optimize.newton(f3,2,fprime=df3)
r1s = optimize.newton(f3,0,x1=0.5)
r2s = optimize.newton(f3,1,x1=2)
print("EJERCICIO 3")
print("Bisección:",r1b,r2b)
print("Newton:",r1n,r2n)
print("Secante:",r1s,r2s)

# -------------------------
# EJERCICIO 4: g(x)=e^{-x}-cos(ax)
a_values = [1,3,5,9]
x4 = np.linspace(0,2,200)
plt.figure(figsize=(6,4))
for a in a_values:
    g = lambda x: np.exp(-x)-np.cos(a*x)
    r = optimize.bisect(g,0,2)
    print(f"EJERCICIO 4: a={a} -> raíz: {r}")
    plt.plot(x4,g(x4), label=f"a={a}")
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 4")
plt.legend()
plt.grid()
plt.show()

# -------------------------
# EJERCICIO 5: g(x)=e^x-log(x+1)-b
b_values=[2,3,5]
x5 = np.linspace(0,3,200)
plt.figure(figsize=(6,4))
for b in b_values:
    g = lambda x: np.exp(x) - np.log(x+1) - b
    r = optimize.newton(g,1)
    print(f"EJERCICIO 5: b={b} -> raíz: {r}")
    plt.plot(x5,g(x5), label=f"b={b}")
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 5")
plt.legend()
plt.grid()
plt.show()

# -------------------------
# EJERCICIO 6: sqrt(x)-sin(x)-c
f6 = lambda x,c: np.sqrt(x)-np.sin(x)-c
df6 = lambda x,c: 1/(2*np.sqrt(x)) - np.cos(x)
x6 = np.linspace(0,40,500)
plt.figure(figsize=(6,4))
for c,color in zip([1,np.pi],["firebrick","steelblue"]):
    plt.plot(x6,f6(x6,c),label=f"c={c}")
    r_b = optimize.bisect(f6,0 if c==1 else 2,5 if c==1 else 7,args=(c,))
    r_n = optimize.newton(f6,4 if c==1 else 3,fprime=df6,args=(c,))
    r_s = optimize.newton(f6,1 if c==1 else 2,x1=4 if c==1 else 4,args=(c,))
    print(f"EJERCICIO 6: c={c} -> Bisección:{r_b}, Newton:{r_n}, Secante:{r_s}")
plt.axhline(0,color="black",linestyle="--")
plt.title("Ejercicio 6")
plt.legend()
plt.grid()
plt.show()

# -------------------------
# EJERCICIO 7: Sistema y curvas
f7_1 = lambda x: np.sqrt(x**2 -4*x +2)
f7_2 = lambda x: -np.sqrt(x**2 -4*x +2)
g7_1 = lambda x: np.sqrt((-x**2 +4)/3)
g7_2 = lambda x: -np.sqrt((-x**2 +4)/3)
x7 = np.linspace(-2,2,500)
plt.figure(figsize=(6,4))
plt.plot(x7,f7_1(x7),label="f1")
plt.plot(x7,f7_2(x7),label="f2")
plt.plot(x7,g7_1(x7),label="g1")
plt.plot(x7,g7_2(x7),label="g2")
plt.axhline(0,color="black",linestyle="--")
plt.axvline(0,color="black",linestyle="--")
plt.title("Ejercicio 7")
plt.legend()
plt.grid()
plt.show()
f7_eq = lambda x:(4/3)*x**2 -4*x +2/3
df7 = lambda x:(8/3)*x -4
x_root = optimize.newton(f7_eq,1)
y1_root = f7_1(x_root)
y2_root = f7_2(x_root)
print("EJERCICIO 7 - Intersecciones:")
print(f"({x_root},{y1_root}), ({x_root},{y2_root})")

# -------------------------
# EJERCICIO 8: Sistema sen(x)+y^2-1=0, x+cos(y)-1=0
def sistema8(vars):
    x,y = vars
    return [np.sin(x)+y**2-1, x+np.cos(y)-1]
sol8 = optimize.fsolve(sistema8,[0.25,1])
print("EJERCICIO 8 - Solución:",sol8)
f8_1 = lambda x: np.sqrt(1-np.sin(x))
f8_2 = lambda x: -f8_1(x)
g8 = lambda x: np.arccos(1-x)
x8 = np.linspace(0,2,200)
plt.figure(figsize=(6,4))
plt.plot(x8,f8_1(x8),label="f1")
plt.plot(x8,f8_2(x8),label="f2")
plt.plot(x8,g8(x8),label="g")
plt.plot(sol8[0],sol8[1], 'ro', label="Solución")
plt.axhline(0,color="black",linestyle="--")
plt.axvline(0,color="black",linestyle="--")
plt.title("Ejercicio 8")
plt.legend()
plt.grid()
plt.show()

# -------------------------
# EJERCICIO 9: Sistema x^3+y-6=0, y^3-x-4=0
def sistema9(vars):
    x,y = vars
    return [x**3 + y -6, y**3 -x -4]
sol9 = optimize.fsolve(sistema9,[1,1])
print("EJERCICIO 9 - Solución:",sol9)

# -------------------------
# EJERCICIO 10: Equilibrio económico
qd = lambda p: 200 -10*p**2
qs = lambda p: 5*p**3 +15
p = np.linspace(0,5,200)
plt.figure(figsize=(6,4))
plt.plot(p,qd(p),label="Demanda")
plt.plot(p,qs(p),label="Oferta")
plt.title("Ejercicio 10")
plt.legend()
plt.grid()
plt.show()
def eq10(vars):
    p,q = vars
    return [q+10*p**2-200, q-5*p**3-15]
sol10 = optimize.fsolve(eq10,[2,100])
print("EJERCICIO 10 - Equilibrio (p,q):",sol10)