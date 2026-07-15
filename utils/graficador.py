import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D 

def graficar_biseccion(f, a, b, iterados, raiz, tolerancia, max_iter):
    """
    Genera la gráfica para el método de Bisección (para Streamlit).
    """
    x_vals = np.linspace(a - 0.5, b + 0.5, 400)
    y_vals = f(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar la función
    ax.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2.5)
    ax.axhline(0, color='black', linewidth=1, alpha=0.8)
    
    # Graficar los puntos iterados con colores progresivos
    n_iter = len(iterados)
    for i, (x, fx) in enumerate(iterados):
        color = plt.cm.viridis(i / max(1, n_iter - 1))
        ax.plot(x, fx, 'o', color=color, markersize=8, 
                label=f'Iteración {i+1}' if i < 10 else '')
        ax.plot([x, x], [0, fx], '--', color=color, alpha=0.3, linewidth=1)
    
    # Graficar la raíz final
    if raiz is not None:
        ax.plot(raiz, f(raiz), 'r*', markersize=15, label=f'Raíz ≈ {raiz:.8f}')
        ax.plot([raiz, raiz], [0, f(raiz)], 'r-', alpha=0.5, linewidth=1)
    
    # Mostrar el intervalo de búsqueda
    ax.axvline(a, color='gray', linestyle=':', alpha=0.7, label=f'a = {a:.4f}')
    ax.axvline(b, color='gray', linestyle=':', alpha=0.7, label=f'b = {b:.4f}')
    
    ax.set_title(f'MÉTODO DE BISECCIÓN\nTolerancia = {tolerancia}, Iteraciones = {n_iter}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    
    return fig


def graficar_newton_secante(f, dominio, iterados, raiz, metodo, tolerancia, max_iter):
    """
    Genera la gráfica para los métodos de Newton o Secante (para Streamlit).
    """
    x_min, x_max = dominio
    if raiz is not None:
        x_min = min(x_min, raiz - 0.5)
        x_max = max(x_max, raiz + 0.5)
    
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = f(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar la función
    ax.plot(x_vals, y_vals, 'b-', label='f(x)', linewidth=2.5)
    ax.axhline(0, color='black', linewidth=1, alpha=0.8)
    
    # Graficar los iterados
    n_iter = len(iterados)
    for i, (x, fx) in enumerate(iterados):
        color = plt.cm.plasma(i / max(1, n_iter - 1))
        ax.plot(x, fx, 'o', color=color, markersize=9, 
                label=f'Iteración {i+1}' if i < 8 else '')
        ax.plot([x, x], [0, fx], '--', color=color, alpha=0.3, linewidth=1)
    
    # Graficar la raíz
    if raiz is not None:
        ax.plot(raiz, f(raiz), 'r*', markersize=15, label=f'Raíz ≈ {raiz:.8f}')
        ax.plot([raiz, raiz], [0, f(raiz)], 'r-', alpha=0.5, linewidth=1)
    
    ax.set_title(f'MÉTODO DE {metodo.upper()}\nTolerancia = {tolerancia}, Iteraciones = {n_iter}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    
    return fig


def graficar_interpolacion(x_datos, y_datos, polinomio, grado, x_original=None, y_original=None, titulo="Interpolación"):
    """
    Grafica la interpolación de Newton o Lagrange (para Streamlit).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar los puntos de datos
    ax.scatter(x_datos, y_datos, color='red', s=100, zorder=5, 
               label='Datos', edgecolors='black', linewidth=1.5)
    
    # Generar puntos para el polinomio interpolador
    x_min, x_max = min(x_datos) - 0.5, max(x_datos) + 0.5
    x_interp = np.linspace(x_min, x_max, 300)
    y_interp = polinomio(x_interp)
    
    # Graficar el polinomio
    ax.plot(x_interp, y_interp, 'b-', linewidth=2.5, 
            label=f'Polinomio Grado {grado}')
    
    # Graficar la función original si se proporciona
    if x_original is not None and y_original is not None:
        ax.plot(x_original, y_original, 'g--', linewidth=2, 
                label='Función Original', alpha=0.7)
    
    ax.set_title(f'{titulo} - Grado {grado}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    
    return fig


def graficar_integracion(f, a, b, n, metodo, resultado):
    """
    Visualiza el área bajo la curva para métodos de integración numérica (para Streamlit).
    """
    # Generar puntos para la función
    x_vals = np.linspace(a - 0.5, b + 0.5, 500)
    y_vals = f(x_vals)
    
    # Puntos para el área bajo la curva
    x_fill = np.linspace(a, b, 200)
    y_fill = f(x_fill)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar la función
    ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.8)
    
    # Rellenar el área bajo la curva
    ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='lightblue', 
                    label=f'Área ≈ {resultado:.6f}')
    
    # Marcar límites de integración
    ax.axvline(a, color='gray', linestyle=':', alpha=0.7, label=f'a = {a:.2f}')
    ax.axvline(b, color='gray', linestyle=':', alpha=0.7, label=f'b = {b:.2f}')
    
    ax.set_title(f'INTEGRACIÓN NUMÉRICA - {metodo}\nn = {n} segmentos, Resultado = {resultado:.8f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=10)
    
    return fig


def graficar_comparacion_errores(errores, iteraciones, metodo, titulo="Evolución del Error"):
    """
    Grafica la evolución del error en las iteraciones (para Streamlit).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(iteraciones, errores, 'b-o', linewidth=2, markersize=8)
    
    ax.set_title(f'{titulo} - {metodo}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteración', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig

def graficar_sistema_2d(A, b, solucion, metodo="Gauss"):
    """
    Grafica un sistema de ecuaciones 2x2 mostrando las rectas y la solución.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extraer coeficientes: a1*x + b1*y = c1, a2*x + b2*y = c2
    a1, b1, c1 = A[0, 0], A[0, 1], b[0]
    a2, b2, c2 = A[1, 0], A[1, 1], b[1]
    
    # Generar puntos para las rectas
    x_vals = np.linspace(-10, 10, 400)
    
    # Graficar ecuación 1
    if abs(b1) > 1e-12:
        y1 = (c1 - a1 * x_vals) / b1
        ax.plot(x_vals, y1, 'b-', linewidth=2.5, 
                label=f'Ec. 1: {a1:.2f}x + {b1:.2f}y = {c1:.2f}')
    else:
        # Recta vertical
        x_const = c1 / a1 if abs(a1) > 1e-12 else 0
        ax.axvline(x_const, color='b', linewidth=2.5, 
                   label=f'Ec. 1: x = {x_const:.2f}')
    
    # Graficar ecuación 2
    if abs(b2) > 1e-12:
        y2 = (c2 - a2 * x_vals) / b2
        ax.plot(x_vals, y2, 'r-', linewidth=2.5, 
                label=f'Ec. 2: {a2:.2f}x + {b2:.2f}y = {c2:.2f}')
    else:
        x_const = c2 / a2 if abs(a2) > 1e-12 else 0
        ax.axvline(x_const, color='r', linewidth=2.5, 
                   label=f'Ec. 2: x = {x_const:.2f}')
    
    # Marcar la solución
    if solucion is not None and len(solucion) >= 2:
        ax.plot(solucion[0], solucion[1], 'g*', markersize=20, 
                label=f'Solución: ({solucion[0]:.4f}, {solucion[1]:.4f})')
        # Agregar círculo alrededor de la solución
        ax.plot(solucion[0], solucion[1], 'go', markersize=30, alpha=0.3)
    
    ax.set_title(f'SISTEMA LINEAL 2x2 - {metodo}', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    return fig


def graficar_sistema_3d(A, b, solucion, metodo="Gauss"):
    """
    Grafica un sistema de ecuaciones 3x3 mostrando los planos y la solución.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear malla para los planos
    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)
    X, Y = np.meshgrid(x, y)
    
    # Colores para los planos
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Rojo, Teal, Azul
    
    # Graficar cada plano
    for i in range(3):
        a1, a2, a3, c = A[i, 0], A[i, 1], A[i, 2], b[i]
        if abs(a3) > 1e-12:
            Z = (c - a1 * X - a2 * Y) / a3
            ax.plot_surface(X, Y, Z, alpha=0.4, color=colors[i], 
                           label=f'Plano {i+1}: {a1:.2f}x + {a2:.2f}y + {a3:.2f}z = {c:.2f}')
    
    # Marcar la solución
    if solucion is not None and len(solucion) >= 3:
        ax.scatter(solucion[0], solucion[1], solucion[2], 
                  color='yellow', s=150, marker='*', edgecolors='black', linewidth=2,
                  label=f'Solución: ({solucion[0]:.4f}, {solucion[1]:.4f}, {solucion[2]:.4f})')
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'SISTEMA LINEAL 3x3 - {metodo}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    # Ajustar límites
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    return fig


def graficar_sistema_lineal(A, b, solucion, metodo="Gauss", n_dim=2):
    """
    Función principal que decide qué tipo de gráfica usar según la dimensión.
    """
    if n_dim == 2:
        return graficar_sistema_2d(A, b, solucion, metodo)
    elif n_dim == 3:
        return graficar_sistema_3d(A, b, solucion, metodo)
    else:
        # Para sistemas de más de 3 dimensiones, graficar una matriz de calor
        return graficar_matriz_calor(A, b, solucion, metodo)


def graficar_matriz_calor(A, b, solucion, metodo="Gauss"):
    """
    Grafica la matriz A como un mapa de calor.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa de calor de la matriz A
    im1 = ax1.imshow(A, cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax1.set_title('Matriz A', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Columnas')
    ax1.set_ylabel('Filas')
    
    # Agregar valores en las celdas
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax1.text(j, i, f'{A[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im1, ax=ax1)
    
    # Vector solución
    if solucion is not None:
        im2 = ax2.bar(range(len(solucion)), solucion, color='#4ECDC4', 
                      edgecolor='black', linewidth=1)
        ax2.set_title('Vector Solución', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Variable')
        ax2.set_ylabel('Valor')
        ax2.set_xticks(range(len(solucion)))
        ax2.set_xticklabels([f'x{i+1}' for i in range(len(solucion))])
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Agregar valores encima de las barras
        for i, bar in enumerate(im2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    fig.suptitle(f'SISTEMA LINEAL {A.shape[0]}x{A.shape[1]} - {metodo}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def graficar_comparativa_sistemas(resultados):
    """
    Grafica comparativa de soluciones de diferentes métodos.
    """
    if not resultados:
        return None
    
    metodos = list(resultados.keys())
    n_vars = len(resultados[metodos[0]]['solucion'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Posiciones para las barras
    x = np.arange(n_vars)
    width = 0.35
    
    # Colores para cada método
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (metodo, data) in enumerate(resultados.items()):
        sol = data['solucion']
        offset = (i - len(resultados)/2 + 0.5) * width
        bars = ax.bar(x + offset, sol, width, label=metodo, color=colors[i % len(colors)])
        
        # Agregar valores
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Variables', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparativa de Soluciones entre Métodos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'x{i+1}' for i in range(n_vars)])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    return fig


def graficar_convergencia_seidel(historial, metodo="Gauss-Seidel"):
    """
    Grafica la evolución de las variables en el método de Gauss-Seidel.
    """
    if not historial:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    historial = np.array(historial)
    n_iter = historial.shape[0]
    n_vars = historial.shape[1]
    
    # Graficar cada variable
    for i in range(n_vars):
        ax.plot(range(1, n_iter + 1), historial[:, i], 
                'o-', linewidth=2, markersize=6, label=f'x{i+1}')
    
    ax.set_xlabel('Iteración', fontsize=12)
    ax.set_ylabel('Valor de la variable', fontsize=12)
    ax.set_title(f'Evolución de Variables - {metodo}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    return fig