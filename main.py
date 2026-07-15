import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importación de los módulos desarrollados por tu equipo
from core.raices import metodo_Newton, metodo_biseccion, metodo_secante
from core.interpolaciones_integracion import (
    interpolacion_newton, 
    interpolacion_lagrange, 
    simpson_integral, 
    trapecio_compuesto
)
from core.sistemas_lineales import Gauss_pivoteo, Gauss_seidel
from utils.graficador import graficar_biseccion, graficar_newton_secante, graficar_comparacion_errores, graficar_interpolacion, graficar_integracion, graficar_sistema_lineal, graficar_convergencia_seidel, graficar_comparativa_sistemas

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

mostrar_graficas = st.sidebar.checkbox("📈 Mostrar gráficas", value=True)

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
                    
                    if mostrar_graficas and len(historial) > 1:
                        st.write("### 📈 Evolución del Error")
                        errores = [d['error'] for d in historial[1:]]
                        iteraciones = list(range(2, len(historial) + 1))
                        fig_error = graficar_comparacion_errores(errores, iteraciones, metodo, "Evolución del Error")
                        st.pyplot(fig_error)
                        plt.close(fig_error)
                    
                    
                    if mostrar_graficas:
                        st.write("### 📊 Visualización del Método")
                        
                        # Crear función para graficar
                        def f_grafica(x):
                            return eval(funcion_str, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, 
                                                     "tan": np.tan, "sqrt": np.sqrt, "exp": np.exp, 
                                                     "log": np.log, "pi": np.pi})
                        
                        # Preparar datos para la gráfica
                        iterados_puntos = [(d['x'], d['fx']) for d in historial]
                        
                        if metodo == "Bisección":
                            fig = graficar_biseccion(
                                f_grafica, param_a, param_b, iterados_puntos, 
                                raiz, global_tol, global_max_iter
                            )
                        else:
                            # Newton o Secante
                            dominio = (min(param_a, param_b, raiz) - 1, max(param_a, param_b, raiz) + 1)
                            fig = graficar_newton_secante(
                                f_grafica, dominio, iterados_puntos, 
                                raiz, metodo.replace("-Raphson", ""), global_tol, global_max_iter
                            )
                        
                        st.pyplot(fig)
                        plt.close(fig)
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
        
        tipo_visualizacion = st.selectbox(
            "Tipo de visualización:",
            ["Geométrica (rectas/planos)", "Mapa de calor", "Ambas"]
        )
        
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
                    # Diccionario para almacenar resultados
                    resultados = {}
                    historial_seidel = []
                    
                    # === GAUSS CON PIVOTEO ===
                    if metodo_sistema in ["Gauss con Pivoteo", "Ambos (comparativa)"]:
                        st.write("### 📐 Gauss con Pivoteo")
                        sol_gauss = Gauss_pivoteo(A_np, b_np)
                        
                        st.success("✅ **Sistema resuelto con éxito**")
                        st.write("**Vector Solución:**")
                        sol_df = pd.DataFrame({
                            'Variable': [f'x{i+1}' for i in range(len(sol_gauss))],
                            'Valor': sol_gauss
                        })
                        st.dataframe(sol_df, use_container_width=True)
                        
                        resultados['Gauss'] = {'solucion': sol_gauss, 'iteraciones': 1}
                        
                        # Gráfica del sistema
                        if mostrar_graficas and n_dim in [2, 3]:
                            st.write("**Visualización Geométrica:**")
                            if tipo_visualizacion in ["Geométrica (rectas/planos)", "Ambas"]:
                                fig_gauss = graficar_sistema_lineal(
                                    A_np, b_np, sol_gauss, 
                                    f"Gauss con Pivoteo", n_dim
                                )
                                st.pyplot(fig_gauss)
                                plt.close(fig_gauss)
                            
                            if tipo_visualizacion in ["Mapa de calor", "Ambas"] and n_dim >= 3:
                                st.write("**Mapa de calor de la Matriz:**")
                                fig_heat = graficar_sistema_lineal(
                                    A_np, b_np, sol_gauss, 
                                    f"Gauss con Pivoteo (Heatmap)", n_dim
                                )
                                st.pyplot(fig_heat)
                                plt.close(fig_heat)
                    
                    # === GAUSS-SEIDEL ===
                    if metodo_sistema in ["Gauss-Seidel", "Ambos (comparativa)"]:
                        st.write("### 🔄 Gauss-Seidel")
                        
                        # Ejecutar Gauss-Seidel guardando historial
                        def gauss_seidel_con_historial(A, b, tol, max_iter):
                            A = np.array(A, dtype=float)
                            b = np.array(b, dtype=float)
                            n = len(A)
                            x = np.zeros(n)
                            historial = []
                            
                            for k in range(max_iter):
                                x_anterior = x.copy()
                                for i in range(n):
                                    suma1 = np.dot(A[i, :i], x[:i])
                                    suma2 = np.dot(A[i, i+1:], x_anterior[i+1:])
                                    x[i] = (b[i] - suma1 - suma2) / A[i, i]
                                
                                historial.append(x.copy())
                                
                                denominador = np.where(np.abs(x) > 1e-12, np.abs(x), 1.0)
                                error_relativo = np.max(np.abs((x - x_anterior) / denominador)) * 100.0
                                
                                if error_relativo < tol:
                                    return x, True, k + 1, historial
                            
                            return x, False, max_iter, historial
                        
                        sol_seidel, convergio, iter_seidel, historial_seidel = gauss_seidel_con_historial(
                            A_np, b_np, global_tol, global_max_iter
                        )
                        
                        if convergio:
                            st.success(f"🎉 **¡Convergió con éxito en {iter_seidel} iteraciones!**")
                        else:
                            st.warning(f"⚠️ **El método no convergió después de {iter_seidel} iteraciones.**")
                        
                        st.write("**Vector Solución:**")
                        sol_df = pd.DataFrame({
                            'Variable': [f'x{i+1}' for i in range(len(sol_seidel))],
                            'Valor': sol_seidel
                        })
                        st.dataframe(sol_df, use_container_width=True)
                        
                        resultados['Gauss-Seidel'] = {'solucion': sol_seidel, 'iteraciones': iter_seidel}
                        
                        # Gráfica de convergencia
                        if mostrar_graficas and historial_seidel:
                            st.write("**📈 Evolución de las variables:**")
                            fig_conv = graficar_convergencia_seidel(historial_seidel, "Gauss-Seidel")
                            if fig_conv:
                                st.pyplot(fig_conv)
                                plt.close(fig_conv)
                        
                        # Gráfica del sistema
                        if mostrar_graficas and n_dim in [2, 3]:
                            st.write("**Visualización Geométrica (Gauss-Seidel):**")
                            fig_seidel = graficar_sistema_lineal(
                                A_np, b_np, sol_seidel, 
                                f"Gauss-Seidel ({iter_seidel} iteraciones)", n_dim
                            )
                            st.pyplot(fig_seidel)
                            plt.close(fig_seidel)
                    
                    # === COMPARATIVA ===
                    if metodo_sistema == "Ambos (comparativa)" and len(resultados) > 1:
                        st.write("### 📊 Comparativa de Métodos")
                        fig_comp = graficar_comparativa_sistemas(resultados)
                        if fig_comp:
                            st.pyplot(fig_comp)
                            plt.close(fig_comp)
                        
                        # Tabla comparativa
                        st.write("**Tabla Comparativa:**")
                        comp_df = pd.DataFrame({
                            'Método': list(resultados.keys()),
                            'Iteraciones': [r['iteraciones'] for r in resultados.values()],
                            'Solución': [r['solucion'] for r in resultados.values()]
                        })
                        st.dataframe(comp_df, use_container_width=True)
                            
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
                            
                            # Mostrar gráficas
                            if mostrar_graficas:
                                st.write("### 📊 Visualización de la Interpolación")
                                
                                # Crear función polinomial para cada grado
                                def crear_polinomio_newton(orden):
                                    def eval_poly(x):
                                        xterm = 1.0
                                        result = fdd[0, 0]
                                        for i in range(1, orden + 1):
                                            xterm *= (x - x_arr[i-1])
                                            result += fdd[0, i] * xterm
                                        return result
                                    return eval_poly
                                
                                # Graficar todos los grados
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(x_arr, y_arr, color='red', s=100, label='Datos', zorder=5)
                                
                                x_plot = np.linspace(min(x_arr)-0.5, max(x_arr)+0.5, 200)
                                colors = plt.cm.viridis(np.linspace(0, 1, len(x_arr)))
                                
                                for i in range(1, len(x_arr)):
                                    poly = crear_polinomio_newton(i)
                                    ax.plot(x_plot, poly(x_plot), '--', color=colors[i], 
                                           linewidth=1.5, label=f'Grado {i}')
                                
                                ax.set_title('Comparativa de Polinomios de Newton por Grado', 
                                            fontsize=14, fontweight='bold')
                                ax.set_xlabel('x', fontsize=12)
                                ax.set_ylabel('y', fontsize=12)
                                ax.grid(True, alpha=0.3)
                                ax.legend(loc='best')
                                
                                st.pyplot(fig)
                                plt.close(fig)
                            
                        elif metodo_interp == "Lagrange":
                            y_final = interpolacion_lagrange(x_arr, y_arr, xi_val)
                            
                            st.success("✅ **Interpolación finalizada**")
                            st.metric(label=f"Valor aproximado f({xi_val})", value=f"{y_final:.6f}")
                            
                            # Gráfica de interpolación de Lagrange
                            if mostrar_graficas:
                                st.write("### 📊 Visualización de la Interpolación")
                                
                                def polinomio_lagrange(x):
                                    resultado = 0
                                    n = len(x_arr)
                                    for i in range(n):
                                        L = 1
                                        for j in range(n):
                                            if i != j:
                                                L *= (x - x_arr[j]) / (x_arr[i] - x_arr[j])
                                        resultado += y_arr[i] * L
                                    return resultado
                                
                                fig = graficar_interpolacion(
                                    x_arr, y_arr, polinomio_lagrange, 
                                    len(x_arr)-1, titulo="Interpolación de Lagrange"
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                            
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

                    if mostrar_graficas:
                        st.write("### 📊 Visualización del Área Bajo la Curva")
                        fig = graficar_integracion(
                            f_int, lim_a, lim_b, int(num_segmentos), 
                            metodo_integrar, resultado
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                except Exception as e:
                    st.error(f"❌ **Error al integrar la función:** {str(e)}")
            else:
                st.info("Presione 'Calcular Área' para evaluar la integral definida.")