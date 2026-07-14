import numpy as np

def Diagonal_dominante(A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_fuera_diagonal = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= suma_fuera_diagonal:
            return False
    return True

def Gauss_pivoteo(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    n = len(A)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    print("\n" + "=" * 70)
    print("METODO DE GAUSS CON PIVOTEO")
    print("=" * 70)
    print("\nMATRIZ AUMENTADA INICIAL:")
    print(np.round(Ab, 4))
    
    for i in range(n):
        print(f"\n--- COLUMNA {i+1} ---")
        fila_max = np.argmax(np.abs(Ab[i:, i])) + i
        if fila_max != i:
            print(f"  Intercambio: f{i+1} ↔ f{fila_max+1}")
            Ab[[i, fila_max]] = Ab[[fila_max, i]]
            print("  Matriz despues del intercambio:")
            print(np.round(Ab, 4))
        else:
            print(f"  No hace falta intercambio.")
        
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            print(f"  f{j+1} → f{j+1} - ({factor:.4f}) * f{i+1}")
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]
        
        print("  Matriz despues de eliminar:")
        print(np.round(Ab, 4))
    
    for i in range(n):
        if abs(Ab[i, i]) < 1e-12:
            raise ValueError("SISTEMA SINGULAR O SIN SOLUCION UNICA.")
    
    print("\n" + "=" * 70)
    print("SUSTITUCION HACIA ATRAS")
    print("=" * 70)
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        x[i] = (Ab[i, -1] - suma) / Ab[i, i]
        print(f"  x{i+1} = {x[i]:.3f}")
    
    print("\n" + "=" * 70)
    print("SOLUCION:")
    print(f"  {np.round(x, 3)}")
    print("=" * 70)
    
    return np.round(x, 3)

def Gauss_seidel(A, b, tol=5.0, max_iter=50, verbose=True):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    if np.linalg.det(A) == 0:
        raise ValueError("MATRIZ SINGULAR. GAUSS-SEIDEL NO ES APLICABLE")
    
    n = len(A)
    x = np.zeros(n)
    
    if verbose:
        print("\n" + "=" * 60)
        print("METODO DE GAUSS-SEIDEL")
        print("=" * 60)
        if Diagonal_dominante(A):
            print(" LA MATRIZ ES DIAGONALMENTE DOMINANTE.")
        else:
            print("LA MATRIZ NO ES DIAGONALMENTE DOMINANTE. PUEDE DIVERGIR")
        print(f"Tolerancia: {tol}%")
        print(f"Maximo de iteraciones: {max_iter}")
        print("-" * 60)
    
    for k in range(max_iter):
        x_anterior = x.copy()
        for i in range(n):
            suma1 = np.dot(A[i, :i], x[:i])
            suma2 = np.dot(A[i, i + 1:], x_anterior[i + 1:])
            x[i] = (b[i] - suma1 - suma2) / A[i, i]
        
        denominador = np.where(np.abs(x) > 1e-12, np.abs(x), 1.0)
        error_relativo = np.max(np.abs((x - x_anterior) / denominador)) * 100.0
        
        if verbose:
            print(f"Iteracion {k+1}: x = [{', '.join(f'{xi:.6f}' for xi in x)}]")
            for i in range(n):
                if abs(x[i]) > 1e-12:
                    val = abs((x[i] - x_anterior[i]) / x[i]) * 100.0
                else:
                    val = 100.0
                print(f"  |e{i+1}| = {val:.2f} %")
            print(f"  Error maximo = {error_relativo:.2f} %\n")
        
        if error_relativo > 1e4:
            if verbose:
                print("DIVERGENCIA: EL ERROR RELATIVO ES MUY GRANDE.")
            return x, False, k + 1
        
        if error_relativo < tol:
            if verbose:
                print(f"CONVERGIO EN {k + 1} ITERACIONES.")
            return x, True, k + 1
    
    if verbose:
        print(f"NO CONVERGIO EN {max_iter} ITERACIONES.")
    return x, False, max_iter