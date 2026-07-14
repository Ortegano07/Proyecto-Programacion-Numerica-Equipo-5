import numpy as np

def es_diagonal_dominante(A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_fuera_diagonal = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= suma_fuera_diagonal:
            return False
    return True

def tiene_solucion_unica(A, b, tol=1e-12):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    rango_A = np.linalg.matrix_rank(A, tol=tol)
    ampliada = np.hstack([A, b.reshape(-1, 1)])
    rango_ampliada = np.linalg.matrix_rank(ampliada, tol=tol)
    
    return rango_A == rango_ampliada == A.shape[0]

import numpy as np

def gauss_eliminacion(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    
    # Creamos la matriz aumentada
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    print("\n--- MATRIZ INICIAL ---")
    print(Ab)
    
    for i in range(n):
        # 1. PIVOTEO: Buscamos la fila con el máximo valor en valor absoluto
        fila_max = np.argmax(np.abs(Ab[i:, i])) + i
        
        # Intercambiamos si es necesario (como en tu cuaderno)
        if fila_max != i:
            print(f"\n🔄 [Pivoteo] Intercambiar Fila {i+1} ↔ Fila {fila_max+1}")
            Ab[[i, fila_max]] = Ab[[fila_max, i]]
            print(Ab)
            
        # 2. ELIMINACIÓN: Hacer ceros abajo
        for j in range(i + 1, n):
            # Tu fórmula del cuaderno: factor = coef_fila_abajo / coef_pivote
            factor = Ab[j, i] / Ab[i, i]
            print(f"   -> Fila {j+1} = Fila {j+1} - ({factor:.4f}) * Fila {i+1}")
            Ab[j, i:] -= factor * Ab[i, i:]
            
        print("\n📉 Matriz resultante del paso:")
        print(Ab)
    
    # 3. SUSTITUCIÓN HACIA ATRÁS: Resolver las variables
    x = np.zeros(n)
    print("\n🔹 SUSTITUCIÓN HACIA ATRÁS:")
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]
        print(f"x_{i+1} = {x[i]:.4f}")
        
    return x

def gauss_seidel(A, b, tol=1e-5, max_iter=150, x0=None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    
    if not tiene_solucion_unica(A, b):
        raise ValueError("❌ El sistema no tiene solución única. Gauss-Seidel no puede continuar.")
    
    if np.any(np.abs(np.diag(A)) < 1e-12):
        raise ValueError("❌ La matriz tiene ceros en la diagonal. Gauss-Seidel no es aplicable.")
    
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)
        if len(x) != n:
            raise ValueError("x0 debe tener la misma dimensión que el sistema.")
    
    if not es_diagonal_dominante(A):
        print("⚠️ Advertencia: La matriz no es diagonalmente dominante. Puede no converger.")
    
    historial = []
    convergio = False
    
    for k in range(max_iter):
        x_anterior = x.copy()
        
        for i in range(n):
            suma1 = np.dot(A[i, :i], x[:i])
            suma2 = np.dot(A[i, i + 1:], x_anterior[i + 1:])
            x[i] = (b[i] - suma1 - suma2) / A[i, i]
        
        denominador = np.where(np.abs(x) > 0, np.abs(x), 1.0)
        error_relativo = np.max(np.abs((x - x_anterior) / denominador)) * 100.0
        historial.append(error_relativo)
        
        if error_relativo < tol:
            convergio = True
            break
    
    return x, np.array(historial), convergio, k + 1

 #============================================
# PRUEBAS CON DIFERENTES SISTEMAS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 PRUEBA 1: Sistema con solución única (matriz diagonalmente dominante)")
    print("=" * 60)
    
    # Sistema 1: Matriz diagonalmente dominante (convergencia asegurada)
    A1 = np.array([[4, -1, 1],
                   [2, 5, 2],
                   [1, -2, 6]], dtype=float)
    b1 = np.array([8, 3, 15], dtype=float)
    
    print("Matriz A:")
    print(A1)
    print("Vector b:", b1)
    
    # Gauss
    print("\n🔹 Gauss:")
    sol1 = gauss_eliminacion(A1, b1)
    print(f"Solución: {sol1}")
    
    # Gauss-Seidel
    print("\n🔹 Gauss-Seidel:")
    sol_gs1, hist1, conv1, iter1 = gauss_seidel(A1, b1, tol=1e-5, max_iter=150)
    print(f"Solución: {sol_gs1}")
    print(f"Convergió: {conv1}")
    print(f"Iteraciones: {iter1}")
    print(f"Error final: {hist1[-1]:.4e}%")
    
    # ========================================
    
    print("\n" + "=" * 60)
    print("🧪 PRUEBA 2: Sistema mal condicionado (pivoteo necesario)")
    print("=" * 60)
    
    # Sistema 2: Matriz con números muy pequeños (requiere pivoteo)
    A2 = np.array([[0.001, 1.0],
                   [1.0,   1.0]], dtype=float)
    b2 = np.array([1.0, 2.0], dtype=float)
    
    print("Matriz A:")
    print(A2)
    print("Vector b:", b2)
    
    # Gauss (con pivoteo, debería funcionar bien)
    print("\n🔹 Gauss:")
    sol2 = gauss_eliminacion(A2, b2)
    print(f"Solución: {sol2}")
    
    # Gauss-Seidel
    print("\n🔹 Gauss-Seidel:")
    sol_gs2, hist2, conv2, iter2 = gauss_seidel(A2, b2, tol=1e-5, max_iter=150)
    print(f"Solución: {sol_gs2}")
    print(f"Convergió: {conv2}")
    print(f"Iteraciones: {iter2}")
    print(f"Error final: {hist2[-1]:.4e}%")
    
    # ========================================
    
    print("\n" + "=" * 60)
    print("🧪 PRUEBA 3: Sistema SIN solución única (debe lanzar error)")
    print("=" * 60)
    
    # Sistema 3: Matriz singular (sin solución única)
    A3 = np.array([[1, 2],
                   [2, 4]], dtype=float)  # Fila 2 = 2 * Fila 1
    b3 = np.array([3, 6], dtype=float)     # Misma proporción (infinitas soluciones)
    
    print("Matriz A:")
    print(A3)
    print("Vector b:", b3)
    
    # Gauss (debe fallar)
    print("\n🔹 Gauss:")
    try:
        sol3 = gauss_eliminacion(A3, b3)
        print(f"Solución: {sol3}")
    except ValueError as e:
        print(f"❌ Error esperado: {e}")
    
    # ========================================
    
    print("\n" + "=" * 60)
    print("🧪 PRUEBA 4: Sistema inconsistente (sin solución)")
    print("=" * 60)
    
    # Sistema 4: Inconsistente
    A4 = np.array([[1, 2],
                   [2, 4]], dtype=float)
    b4 = np.array([3, 7], dtype=float)     # No es múltiplo (sin solución)
    
    print("Matriz A:")
    print(A4)
    print("Vector b:", b4)
    
    # Gauss (debe fallar)
    print("\n🔹 Gauss:")
    try:
        sol4 = gauss_eliminacion(A4, b4)
        print(f"Solución: {sol4}")
    except ValueError as e:
        print(f"❌ Error esperado: {e}")