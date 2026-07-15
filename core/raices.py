import numpy as np
import sympy as sp


#Metodo de Newton
def metodo_Newton(funcion_str, x0, tol, max_ite=150):
    x_simb = sp.Symbol('x')
    try:
        f_expre = sp.sympify(funcion_str)
        df_expre = sp.diff(f_expre, x_simb)
    except Exception:
        return None, [], f"Error al procesar la función, no es valida."
    
   
    f = sp.lambdify(x_simb, f_expre, 'numpy') 
    df = sp.lambdify(x_simb, df_expre, 'numpy')
    
    
    Guardar_Valor = []
    x_actual = float(x0)
    msj = f"Se alcanzó el número máximo de las iteraciones {max_ite}"
    raiz = None
    
   
    for i in range(1, max_ite + 1):
        fx = f(x_actual)
        dfx = df(x_actual)
    
       
        if abs(dfx) < 1e-12:
            return None, Guardar_Valor, f"La derivada es cero. No se puede continuar."
        
        
        x_nuevo = x_actual - (fx / dfx)
        if i == 1:
            e_relativo = 100.0
        else:
            if x_nuevo != 0:
                e_relativo = abs((x_nuevo - x_actual) / x_nuevo)
            else:
                e_relativo = 0.0
            
       
        Guardar_Valor.append({
            "iteracion": i, 
            "x": float(x_nuevo),
            "fx": float(fx),
            "dfx": float(dfx), 
            "error": float(e_relativo)
        })
    
       
        if abs(fx) <= tol:
            raiz = x_nuevo
            msj = f"converge exitosamente : |f(x)| <= {tol}"
            break
        
        if i > 1 and e_relativo <= tol:
            raiz = x_nuevo
            msj = f"converge exitosamente : |error relativo| <= {tol}"
            break
        
        x_actual = x_nuevo

    if raiz is None:
        raiz = x_actual
        
    return raiz, Guardar_Valor, msj
    
    
    
    
#Metodo de biseccion
def metodo_biseccion(funcion_str, a, b, tol, max_ite=150):
    
   
    f = lambda x: eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "pi": np.pi,
                            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan})
   
    Guardar_Valor=[]
    if f(a) * f(b) >= 0:
        return None, [], "La funcion no tiene signos opuestos en el intervalo [a,b]. No se puede garantizar una raiz"
    
    
    x_anterior=a
    msj= f"Se alcanzó el número máximo de las iteraciones {max_ite}"
    raiz = None
    
    
    for i in range(1,max_ite+1):
       
        x_m = (a + b) / 2.0
        fx_m = f(x_m)
        
       
        if i==1:
            e_relativo=100.0
        else:
            if x_m != 0:
                e_relativo = abs((x_m - x_anterior) / x_m)
            else:
                e_relativo = 0.0
        
        Guardar_Valor.append({
            "iteracion": i,
            "x": float(x_m),
            "fx": float(fx_m),
            "error": float(e_relativo) })
        
        if abs(fx_m) <= tol:
            raiz = x_m
            msj = f"converge exitosamente : |f(x_m)| <= {tol}"
            break
            
        if i > 1 and e_relativo <= tol:
            raiz = x_m
            msj = f"converge exitosamente : |error relativo| <= {tol}"
            break
        
        
        if f(a) * fx_m < 0:
            b = x_m
        else:
            a = x_m
        x_anterior = x_m
        
        
    if raiz is not None:
        raiz= x_m
            
    return raiz, Guardar_Valor, msj
        
        
  
#Metodo de la Secante
def metodo_secante(funcion_str, x0, x1, tol, max_ite=150):
    x_simb = sp.Symbol('x')
    try:
        f_expre = sp.sympify(funcion_str)
    except Exception:
        return None, [], f"Error al procesar la función, no es valida."
    
    f = sp.lambdify(x_simb, f_expre, 'numpy')
    
    
    Guardar_Valor = []
    x_ant = float(x0)     
    x_act = float(x1)     
    msj = f"Se alcanzó el número máximo de las iteraciones {max_ite}"
    raiz = None
    
    for i in range(1, max_ite + 1):
        f_ant = f(x_ant)
        f_act = f(x_act)
        
       
        if abs(f_act - f_ant) < 1e-12:
            return None, Guardar_Valor, f"Error: División por cero detectada."
        
        
        x_nuevo = x_act - (f_act * (x_act - x_ant)) / (f_act - f_ant)
        
        
        if x_nuevo != 0:
            e_relativo = abs((x_nuevo - x_act) / x_nuevo)
        else:
            e_relativo = 0.0
            
        
        Guardar_Valor.append({
            "iteracion": i,
            "x": float(x_nuevo),
            "fx": float(f_act),
            "error": float(e_relativo)
        })
        
        
        if abs(f(x_nuevo)) <= tol:
            raiz = x_nuevo
            msj = f"converge exitosamente : |f(x)| <= {tol}"
            break
            
        if e_relativo <= tol:
            raiz = x_nuevo
            msj = f"converge exitosamente : |error relativo| <= {tol}"
            break
            
        
        x_ant = x_act
        x_act = x_nuevo
        
    if raiz is None:
        raiz = x_act
        
    return raiz, Guardar_Valor, msj
    
    