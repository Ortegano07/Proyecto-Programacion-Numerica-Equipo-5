import matplotlib.pyplot as plt
import numpy as np

def graficar_biseccion(f, a, b, iterados, raiz, tolerancia, max_iter):
    """
    Genera la gráfica para el método de Bisección.
    
    Parámetros:
    - f: función lambda o definida.
    - a, b: intervalo de búsqueda inicial.
    - iterados: lista de tuplas (x, f(x)) generadas en cada iteración.
    - raiz: valor de la raíz aproximada.
    - tolerancia: tolerancia usada.
    - max_iter: número máximo de iteraciones.
    """
    # Crear un rango de x para graficar la función en el intervalo de búsqueda
    x_vals = np.linspace(a - 0.5, b + 0.5, 400)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    
    # Graficar la función
    plt.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2)
    
    # Graficar la línea y=0
    plt.axhline(0, color='black', linewidth=0.8)
    
    # Graficar los puntos iterados
    for i, (x, fx) in enumerate(iterados):
        plt.plot(x, fx, 'go', markersize=6, label=f'Iteración {i+1}' if i < 10 else '')
    
    # Graficar la raíz final
    plt.plot(raiz, f(raiz), 'r*', markersize=12, label=f'Raíz ≈ {raiz:.6f}')
    
    # Configurar la gráfica
    plt.title(f'Método de Bisección\nTolerancia = {tolerancia}, Iteraciones = {len(iterados)}', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Mostrar el intervalo de búsqueda
    plt.axvline(a, color='gray', linestyle=':', alpha=0.5, label=f'a = {a}')
    plt.axvline(b, color='gray', linestyle=':', alpha=0.5, label=f'b = {b}')
    
    plt.tight_layout()
    plt.show()

def graficar_newton_secante(f, dominio, iterados, raiz, metodo, tolerancia, max_iter):
    """
    Genera la gráfica para los métodos de Newton o Secante.
    
    Parámetros:
    - f: función lambda o definida.
    - dominio: tupla (x_min, x_max) definida por el usuario.
    - iterados: lista de tuplas (x, f(x)).
    - raiz: valor de la raíz aproximada.
    - metodo: string, 'Newton' o 'Secante'.
    - tolerancia: tolerancia usada.
    - max_iter: número máximo de iteraciones.
    """
    x_min, x_max = dominio
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(10, 6))
    
    # Graficar la función
    plt.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2)
    plt.axhline(0, color='black', linewidth=0.8)
    
    # Graficar los iterados
    for i, (x, fx) in enumerate(iterados):
        plt.plot(x, fx, 'go', markersize=6, label=f'Iteración {i+1}' if i < 5 else '')
    
    # Graficar la raíz
    plt.plot(raiz, f(raiz), 'r*', markersize=12, label=f'Raíz ≈ {raiz:.6f}')
    
    # Configurar la gráfica
    plt.title(f'Método de {metodo}\nTolerancia = {tolerancia}, Iteraciones = {len(iterados)}', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_interpolacion(x_datos, y_datos, polinomio, grado, x_original=None, y_original=None):
    """
    Grafica la interpolación de Newton o Lagrange.
    
    Parámetros:
    - x_datos, y_datos: puntos a interpolar.
    - polinomio: función del polinomio interpolador.
    - grado: grado del polinomio.
    - x_original, y_original: (opcional) función original si se conoce.
    """
    plt.figure(figsize=(10, 6))
    
    # Graficar los puntos de datos
    plt.scatter(x_datos, y_datos, color='red', s=80, label='Datos')
    
    # Generar puntos para el polinomio interpolador
    x_interp = np.linspace(min(x_datos)-0.5, max(x_datos)+0.5, 200)
    y_interp = polinomio(x_interp)
    
    # Graficar el polinomio
    plt.plot(x_interp, y_interp, 'b-', label=f'Polinomio Grado {grado}')
    
    # Graficar la función original si se proporciona
    if x_original is not None and y_original is not None:
        plt.plot(x_original, y_original, 'g--', label='Función Original', alpha=0.7)
    
    plt.title(f'Interpolación de Grado {grado}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()