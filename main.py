import numpy as np
from utils.graficador import graficar_biseccion, graficar_newton_secante

# --- Ejemplo de Bisección ---
def f(x):
    return x**3 - x - 2  # Función de prueba

# Simulación de iterados (esto lo genera el método de bisección)
iterados_biseccion = [(1.0, -2.0), (1.5, -0.125), (1.75, 0.859), (1.625, 0.266), (1.5625, 0.031)]
raiz_biseccion = 1.5625
a, b = 1.0, 2.0

# Llamar a la función graficadora
graficar_biseccion(f, a, b, iterados_biseccion, raiz_biseccion, 0.001, 10)

# --- Ejemplo de Newton ---
def df(x):  # Derivada para Newton
    return 3*x**2 - 1

# Simulación de iterados de Newton
iterados_newton = [(2.0, 4.0), (1.545, 0.145), (1.521, 0.003)]
raiz_newton = 1.521
dominio = (0.5, 2.5)  # Dominio que el usuario ingresa

graficar_newton_secante(f, dominio, iterados_newton, raiz_newton, 'Newton', 0.001, 10)