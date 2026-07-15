import streamlit as st
import pandas as pd
import numpy as np

# Importación de los módulos desarrollados por tu equipo
from core.raices import metodo_Newton, metodo_biseccion, metodo_secante
from core.interpolaciones_integracion import (
    interpolacion_newton, 
    interpolacion_lagrange, 
    simpson_integral, 
    trapecio_compuesto
)
from core.sistemas_lineales import Gauss_pivoteo, Gauss_seidel

# Configuración de página premium
st.set_page_config(
    page_title="Sistema de Soluciones Numéricas", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo personalizado para un look limpio y moderno
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# Cabecera principal
st.markdown('<div class="main-title">📊 Sistema de Soluciones Numéricas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Plataforma académica interactiva para el análisis de algoritmos — UCLA</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# BARRA LATERAL: PARÁMETROS GLOBALES (Criterios de parada)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros Globales")
st.sidebar.write("Defina los criterios de parada comunes para los métodos iterativos.")

# Tolerancia global (convertida a flotante)
tol_input = st.sidebar.text_input("Tolerancia de error (Tol):", value="0.001")
try:
    global_tol = float(tol_input)
except ValueError:
    global_tol = 0.001
    st.sidebar.error("Tolerancia no válida. Usando 0.001 por defecto.")

# Iteraciones máximas globales
global_max_iter = st.sidebar.number_input(
    "Iteraciones máximas:", 
    min_value=1, 
    max_value=500, 
    value=150, 
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Consejo: Los módulos matemáticos en `core/` procesarán automáticamente los datos según las variables ingresadas.")

# ---------------------------------------------------------
# NAVEGACIÓN PRINCIPAL POR PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🧮 Cálculo de Raíces", 
    "📐 Sistemas de Ecuaciones", 
    "📈 Interpolación e Integración"
])

# =========================================================
# PESTAÑA 1: CÁLCULO DE RAÍCES
# =========================================================
with tab1:
    st.header("Resolución de Ecuaciones No Lineales")
    st.write("Seleccione un método para aproximar la raíz de una función $f(x) = 0$.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuración del Problema")
        # Entrada de la función
        funcion_str = st.text_input("Función f(x) en formato Python / SymPy:", value="x**2 - 4")
        
        # Selección de método
        metodo = st.selectbox(
            "Método de resolución:",
            ["Bisección", "Newton-Raphson", "Secante"]
        )
        
        # Parámetros específicos según el método seleccionado
        if metodo == "Bisección":
            st.write("**Intervalo de búsqueda [a, b]**")
            param_a = st.number_input("Límite inferior (a):", value=0.0)
            param_b = st.number_input("Límite superior (b):", value=3.0)
            
        elif metodo == "Newton-Raphson":
            st.write("**Aproximación Inicial**")
            param_x0 = st.number_input("Valor inicial (x0):", value=1.0)
            
        elif metodo == "Secante":
            st.write("**Aproximación Inicial Dual**")
            param_x0 = st.number_input("Valor inicial (x0):", value=1.0)
            param_x1 = st.number_input("Valor inicial (x1):", value=3.0)
            
        # Botón para ejecutar
        ejecutar_raices = st.button("🚀 Calcular Raíz", use_container_width=True)

    with col2:
        st.subheader("Resultados y Convergencia")
        
        if ejecutar_raices:
            raiz = None
            historial = []
            mensaje = ""
            
            # Ejecución según método seleccionado
            with st.spinner("Ejecutando algoritmo matemático..."):
                if metodo == "Bisección":
                    raiz, historial, mensaje = metodo_biseccion(
                        funcion_str, param_a, param_b, global_tol, global_max_iter
                    )
                elif metodo == "Newton-Raphson":
                    raiz, historial, mensaje = metodo_Newton(
                        funcion_str, param_x0, global_tol, global_max_iter
                    )
                elif metodo == "Secante":
                    raiz, historial, mensaje = metodo_secante(
                        funcion_str, param_x0, param_x1, global_tol, global_max_iter
                    )
            
            # Desplegar los resultados de manera elegante
            if raiz is not None:
                st.success(f"**Resultado:** {mensaje}")
                st.metric(label="Raíz Aproximada", value=f"{raiz:.8f}")
                
                # Mostrar la tabla de iteraciones si se guardaron valores
                if historial:
                    st.write("### Historial de Iteraciones")
                    df_iteraciones = pd.DataFrame(historial)
                    # Formateamos la tabla para que se vea impecable
                    st.dataframe(df_iteraciones.style.format({
                        'x': '{:.6f}',
                        'fx': '{:.6e}',
                        'dfx': '{:.6e}', # Solo Newton lo incluye
                        'error': '{:.6f}'
                    }), use_container_width=True)
            else:
                st.error(f"**Error en la ejecución:** {mensaje}")
        else:
            st.info("Presione el botón 'Calcular Raíz' para procesar los datos ingresados.")

# =========================================================
# PESTAÑA 2: SISTEMAS DE ECUACIONES LINEALES
# =========================================================
with tab2:
    st.header("Solución de Sistemas de Ecuaciones Lineales")
    st.write("Resuelva sistemas de la forma $Ax = b$ usando métodos directos e iterativos.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuración de la Matriz")
        
        # Selección del tamaño del sistema (N x N)
        n_dim = st.number_input("Dimensión del sistema (N):", min_value=2, max_value=5, value=3, step=1)
        
        st.write("**Ingrese los coeficientes de la Matriz A:**")
        # Generar inputs dinámicos en cuadrícula para la matriz A
        matriz_A = []
        for i in range(n_dim):
            fila = []
            cols_inputs = st.columns(n_dim)
            for j in range(n_dim):
                # Valor por defecto 1.0 en la diagonal para evitar singularidad inicial fácil
                default_val = 1.0 if i == j else 0.0
                val = cols_inputs[j].number_input(f"A[{i+1},{j+1}]", value=default_val, key=f"A_{i}_{j}")
                fila.append(val)
            matriz_A.append(fila)
            
        st.write("**Ingrese el Vector de resultados b:**")
        # Inputs dinámicos para el vector b
        vector_b = []
        cols_b = st.columns(n_dim)
        for i in range(n_dim):
            val_b = cols_b[i].number_input(f"b[{i+1}]", value=1.0, key=f"b_{i}")
            vector_b.append(val_b)
            
        metodo_sistema = st.selectbox(
            "Método de solución lineal:",
            ["Gauss con Pivoteo", "Gauss-Seidel"]
        )
        
        ejecutar_sistemas = st.button("🚀 Resolver Sistema", use_container_width=True)

    with col2:
        st.subheader("Solución Analítica")
        
        if ejecutar_sistemas:
            A_np = np.array(matriz_A, dtype=float)
            b_np = np.array(vector_b, dtype=float)
            
            with st.spinner("Procesando matriz..."):
                try:
                    if metodo_sistema == "Gauss con Pivoteo":
                        # El método original imprime mucho en terminal, capturamos su salida
                        solucion = Gauss_pivoteo(A_np, b_np)
                        
                        st.success("✅ **Sistema resuelto con éxito (Método Directo)**")
                        st.write("### Vector Solución $x$:")
                        for idx, x_val in enumerate(solucion):
                            st.metric(label=f"Variable x{idx+1}", value=f"{x_val:.4f}")
                            
                    elif metodo_sistema == "Gauss-Seidel":
                        # Llamamos al algoritmo con los parámetros globales de la barra lateral
                        solucion, convergio, iters = Gauss_seidel(
                            A_np, b_np, tol=global_tol, max_iter=global_max_iter, verbose=False
                        )
                        
                        if convergio:
                            st.success(f"🎉 **¡Convergió con éxito en {iters} iteraciones!**")
                        else:
                            st.warning(f"⚠️ **El método no convergió después de {iters} iteraciones o divergió.**")
                            
                        st.write("### Vector Solución $x$:")
                        for idx, x_val in enumerate(solucion):
                            st.metric(label=f"Variable x{idx+1}", value=f"{x_val:.6f}")
                            
                except Exception as e:
                    st.error(f"❌ **Error al resolver el sistema:** {str(e)}")
        else:
            st.info("Configure su matriz y presione 'Resolver Sistema' para observar los cálculos.")


# =========================================================
# PESTAÑA 3: INTERPOLACIÓN E INTEGRACIÓN
# =========================================================
with tab3:
    st.header("Herramientas de Interpolación e Integración")
    
    subtab_interp, subtab_integ = st.tabs(["📌 Interpolación de Puntos", "📐 Integración Numérica"])
    
    # ---------------------------------------------------------
    # SUB-PESTAÑA A: INTERPOLACIÓN
    # ---------------------------------------------------------
    with subtab_interp:
        col_int1, col_int2 = st.columns([1, 2])
        
        with col_int1:
            st.subheader("Puntos de Control")
            st.write("Defina los vectores $X$ y $Y$ (mismo tamaño, separados por comas).")
            
            x_puntos_str = st.text_input("Puntos X:", value="1, 2, 3, 4")
            y_puntos_str = st.text_input("Puntos Y:", value="2, 4, 8, 16")
            
            xi_val = st.number_input("Valor a interpolar (Xi):", value=2.5)
            
            metodo_interp = st.selectbox(
                "Algoritmo de Interpolación:",
                ["Newton (Diferencias Divididas)", "Lagrange"]
            )
            
            ejecutar_interp = st.button("🚀 Calcular Interpolación", use_container_width=True)
            
        with col_int2:
            st.subheader("Resultado de Interpolación")
            
            if ejecutar_interp:
                try:
                    # Parsear los strings a arrays de numpy
                    x_arr = np.array([float(val.strip()) for val in x_puntos_str.split(",")])
                    y_arr = np.array([float(val.strip()) for val in y_puntos_str.split(",")])
                    
                    if len(x_arr) != len(y_arr):
                        st.error("Error: Los vectores X e Y deben tener la misma cantidad de elementos.")
                    else:
                        if metodo_interp == "Newton (Diferencias Divididas)":
                            y_final, y_por_orden, ea, fdd = interpolacion_newton(x_arr, y_arr, xi_val)
                            
                            st.success("✅ **Interpolación finalizada**")
                            st.metric(label=f"Valor aproximado f({xi_val})", value=f"{y_final:.6f}")
                            
                            # Mostrar tabla de diferencias divididas para inspección (un toque pro)
                            st.write("### Tabla de Diferencias Divididas:")
                            st.dataframe(pd.DataFrame(fdd), use_container_width=True)
                            
                        elif metodo_interp == "Lagrange":
                            y_final = interpolacion_lagrange(x_arr, y_arr, xi_val)
                            
                            st.success("✅ **Interpolación finalizada**")
                            st.metric(label=f"Valor aproximado f({xi_val})", value=f"{y_final:.6f}")
                            
                except Exception as e:
                    st.error(f"❌ **Error en los datos de entrada:** {str(e)}")
            else:
                st.info("Ingrese los puntos coordenados y ejecute el cálculo.")
                
    # ---------------------------------------------------------
    # SUB-PESTAÑA B: INTEGRACIÓN
    # ---------------------------------------------------------
    with subtab_integ:
        col_itg1, col_itg2 = st.columns([1, 2])
        
        with col_itg1:
            st.subheader("Configuración de la Integral")
            func_integral_str = st.text_input("Función f(x) a integrar:", value="x**2")
            
            st.write("**Límites de Integración**")
            lim_a = st.number_input("Límite inferior (a):", value=0.0, key="int_a")
            lim_b = st.number_input("Límite superior (b):", value=2.0, key="int_b")
            
            num_segmentos = st.number_input(
                "Número de segmentos (n):", 
                min_value=1, 
                max_value=1000, 
                value=4, 
                step=1
            )
            
            metodo_integrar = st.selectbox(
                "Método de Integración:",
                ["Trapecio Compuesto", "Simpson (Integrador General)"]
            )
            
            ejecutar_integracion = st.button("🚀 Calcular Área", use_container_width=True)
            
        with col_itg2:
            st.subheader("Resultado de la Integración")
            
            if ejecutar_integracion:
                try:
                    # Crear una función ejecutable de forma segura usando lambda y eval controlado
                    f_int = lambda x: eval(func_integral_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sqrt": np.sqrt, "exp": np.exp, "log": np.log})
                    
                    if metodo_integrar == "Trapecio Compuesto":
                        resultado, h_paso = trapecio_compuesto(f_int, lim_a, lim_b, int(num_segmentos))
                    elif metodo_integrar == "Simpson (Integrador General)":
                        resultado, h_paso = simpson_integral(f_int, lim_a, lim_b, int(num_segmentos))
                        
                    st.success("✅ **Integración calculada exitosamente**")
                    st.metric(label="Área aproximada (Integral)", value=f"{resultado:.8f}")
                    st.caption(f"Tamaño de paso utilizado ($h$): {h_paso:.6f}")
                    
                except Exception as e:
                    st.error(f"❌ **Error al integrar la función:** {str(e)}")
            else:
                st.info("Presione 'Calcular Área' para evaluar la integral definida.")