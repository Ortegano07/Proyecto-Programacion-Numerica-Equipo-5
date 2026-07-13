import numpy as np

def metodo_biseccion(funcion_str, a, b, tol, max_ite=150):
    
    #transformar el string en una funcion de python
    f = lambda x: eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "pi": np.pi,
                            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan})
    #teorema de biseccion
    Guardar_Valor=[]
    if f(a) * f(b) >= 0:
        return None, [], "La funcion no tiene signos opuestos en el intervalo [a,b]. No se puede garantizar una raiz"
    
    #Guardar los valores anteriores
    x_anterior=a
    msj= f"Se alcanzó el número máximo de las iteraciones {max_ite}"
    raiz = None
    
    #Ciclo de iteraciones
    for i in range(1,max_ite+1):
        #Punto medio
        x_m = (a + b) / 2.0
        fx_m = f(x_m)
        
        #Calcular el error relativo aprox
        if i==1:
            e_relativo=100.0
        else:
            if x_m != 0:
                e_relativo = abs((x_m - x_anterior) / x_m)
            else:
                e_relativo = 0.0
        #Guardar los datos anteriores para el manejo del frontend y graficas
        Guardar_Valor.append({
            "iteracion": i,
            "x": float(x_m),
            "fx": float(fx_m),
            "error": float(e_relativo) })
        #Criterio de parada combinado
            #Se encontro la raiz
        if abs(fx_m) <= tol:
            raiz = x_m
            msj = f"converge exitosamente : |f(x_m)| <= {tol}"
            break
            #Se estabilizó el metodo
        if i > 1 and e_relativo <= tol:
            raiz = x_m
            msj = f"converge exitosamente : |e_relativo| <= {tol}"
            break
        
        #Actualizar el intervalo
        if f(a) * fx_m < 0:
            b = x_m
        else:
            a = x_m
        x_anterior = x_m
        
        #retorno del resultado
    if raiz is not None:
        raiz= x_m
            
    return raiz, Guardar_Valor, msj
        
    # =============================================================================
# BLOQUE DE PRUEBA LOCAL (Solo se ejecuta si corres este archivo directamente)
# =============================================================================
if __name__ == "__main__":
    print("🧪 PROBANDO EL MÉTODO DE BISECCIÓN...\n")
    
    # Datos de prueba: f(x) = x^2 - 4. Sabemos que las raíces son 2 y -2.
    # Buscaremos en el intervalo [0, 3] con una tolerancia de 0.001
    funcion_prueba = "x**2 - 4"
    limite_inferior = 0.0
    limite_superior = 3.0
    tolerancia = 1e-3
    
    # Llamamos a tu función
    raiz_hallada, tabla_iteraciones, motivo = metodo_biseccion(
        funcion_prueba, 
        limite_inferior, 
        limite_superior, 
        tolerancia
    )
    
    # Mostrar el resultado final
    print(f"Resultado: {motivo}")
    print(f"Raíz aproximada: {raiz_hallada}\n")
    
    # Imprimir la tabla de iteraciones en la terminal
    print(f"{'Iteración':<10}{'x_m (Raíz)':<15}{'f(x_m)':<15}{'Error Relativo':<15}")
    print("-" * 55)
    for fila in tabla_iteraciones:
        print(f"{fila['iteracion']:<10}{fila['x']:<15.5f}{fila['fx']:<15.5f}{fila['error']:<15.5f}")