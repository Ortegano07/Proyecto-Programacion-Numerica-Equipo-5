import numpy as np

def Diagonal_dominante (A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        suma_fuera_diagonal = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= suma_fuera_diagonal:
            return False
    return True

def Tiene_solucion_unica (A, b, tol=1e-6):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada.")
    
    rango_A = np.linalg.matrix_rank(A, tol=tol)
    ampliada = np.hstack([A, b.reshape(-1, 1)])
    rango_ampliada = np.linalg.matrix_rank(ampliada, tol=tol)
    
    return rango_A == rango_ampliada == A.shape[0]


def Gauss_pivoteo(A, b):
   
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)

    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    print("MATRIZ AUMENTADA INICIAL")
    print(Ab)
        
    for i in range(n):
        print(f"\n---------------------------------------------")
        print(f"COLUMNA {i+1}")
        print(f"---------------------------------------------")
       
        fila_max = np.argmax(np.abs(Ab[i:, i])) + i
        
        if fila_max != i:
            print(f"(f{i+1} ↔ f{fila_max+1})")
            Ab[[i, fila_max]] = Ab[[fila_max, i]]
            print("Matriz resultante del intercambio:")
            print(Ab)
        else:
            print(f"No hace falta intercambiar filas (el pivote f{i+1} ya es mayor)")
            
        pivote = Ab[i, i]
        
        for j in range(i + 1, n):
            coeficiente = Ab[j, i]
            factor = coeficiente / pivote
            
            print(f" f{j+1} → f{j+1} - ({coeficiente} / {pivote}) * f{i+1}")
            
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]
            
        print("\n Matriz resultante después de hacer ceros en esta columna:")
        print(Ab)
        
    print("\n=============================================")
    print("        SUSTITUCIÓN HACIA ATRÁS")
    print("=============================================")

    for r in range(n):
        
        elementos_A = " ".join(f"{Ab[r, c]:8.4f}" for c in range(n))
        termino_b = f"{Ab[r, -1]:8.4f}"
        
        print(f"[ {elementos_A}  |  {termino_b} ]  X{r+1}")
        
    print("="*45 + "\n")

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma_conocidos = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        x[i] = (Ab[i, -1] - suma_conocidos) / Ab[i, i]
        print(f"x{i+1} = ({Ab[i, -1]} - ({suma_conocidos:.4f})) / {Ab[i, i]} = {x[i]:.4f}")
        
    return x