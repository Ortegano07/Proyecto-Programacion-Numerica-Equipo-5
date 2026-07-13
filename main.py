import streamlit as st

# 1. Configuración de página con Layout Amplio
st.set_page_config(
    page_title="Numerical Core | UCLA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inyección de CSS Avanzado para pestañas y diseño oscuro
st.markdown("""
    <style>
        /* Fondo general y fuentes */
        .main {
            background-color: #0f172a;
            color: #f8fafc;
        }
        /* Estilizar la barra lateral */
        [data-testid="stSidebar"] {
            background-color: #1e293b !important;
            border-right: 1px solid #334155;
        }
        /* Títulos principales */
        h1 {
            color: #3b82f6 !important;
            font-family: 'Segoe UI', sans-serif;
            font-weight: 800 !important;
            letter-spacing: -1px;
        }
        /* Estilizar las pestañas (Tabs) de Streamlit */
        button[data-baseweb="tab"] {
            font-size: 14px !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            border-bottom: 2px solid transparent !important;
            padding: 10px 20px !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #3b82f6 !important;
            border-bottom: 2px solid #3b82f6 !important;
            background-color: rgba(59, 130, 246, 0.05) !important;
        }
        /* Estilizar botones */
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        }
    </style>
""", unsafe_with_html=True)

# 3. Encabezado principal del Dashboard
st.markdown("# ⚡ NUMERICAL CORE <span style='color:#64748b; font-size:14px;'>v1.0</span>", unsafe_with_html=True)
st.markdown("<p style='color:#94a3b8; font-size:16px; margin-top:-15px;'>Terminal de Cómputo Avanzado y Análisis Numérico — UCLA</p>", unsafe_with_html=True)
st.markdown("---")

# 4. Barra Lateral de Parámetros Globales (Sidebar)
st.sidebar.markdown("<h2 style='color:#f8fafc; font-size:18px; font-weight:700; margin-bottom:20px;'>⚙️ CONFIGURACIÓN GLOBAL</h2>", unsafe_with_html=True)
tolerancia_global = st.sidebar.number_input("Tolerancia del Error (Tol):", value=1e-5, format="%.5e")
max_iteraciones = st.sidebar.number_input("Iteraciones Máximas:", min_value=1, max_value=500, value=100)

# 5. Creación de las PESTAÑAS PRINCIPALES en el centro
tab1, tab2, tab3 = st.tabs([
    "🎯 Raíces de Funciones", 
    "🧮 Sistemas de Ecuaciones Lineales", 
    "📈 Interpolación e Integración"
])

# ==========================================
# PESTAÑA 1: RAÍCES DE FUNCIONES
# ==========================================
with tab1:
    st.markdown("<br>", unsafe_with_html=True)
    # Sub-selección del método específico dentro de la pestaña usando un selector limpio
    metodo_raices = st.selectbox(
        "Seleccione el Algoritmo de Cálculo:",
        ["Bisección", "Newton-Raphson", "Secante"]
    )
    
    # Contenedor para inputs de datos
    st.markdown("<div style='background-color:#1e293b; padding:25px; border-radius:12px; border:1px solid #334155;'>", unsafe_with_html=True)
    col1, col2 = st.columns(2)
    with col1:
        funcion_str = st.text_input("Función Matemática f(x):", "x**2 - 4", key="func_raices")
    with col2:
        if metodo_raices == "Bisección":
            sub_col1, sub_col2 = st.columns(2)
            lim_inf = sub_col1.number_input("Límite Inferior (a):", value=0.0)
            lim_sup = sub_col2.number_input("Límite Superior (b):", value=3.0)
        elif metodo_raices == "Newton-Raphson":
            x0 = st.number_input("Aproximación Inicial (x0):", value=1.0)
        elif metodo_raices == "Secante":
            sub_col1, sub_col2 = st.columns(2)
            x0 = sub_col1.number_input("Aproximación Inicial (x0):", value=1.0)
            x1 = sub_col2.number_input("Aproximación Secundaria (x1):", value=2.0)
    st.markdown("</div>", unsafe_with_html=True)
    
    st.markdown("<br>", unsafe_with_html=True)
    
    if st.button(f"EJECUTAR {metodo_raices.upper()}"):
        st.subheader("📊 Resultados de la Simulación")
        col_res, col_graf = st.columns([5, 7])
        with col_res:
            st.markdown("<p style='color:#10b981; font-weight:600;'>✓ Algoritmo Convergió Exitosamente</p>", unsafe_with_html=True)
            st.code(f"Procesando {metodo_raices}...\nRaíz aproximada encontrada.\nIteraciones utilizadas: 14")
        with col_graf:
            st.markdown("<div style='background-color:#1e293b; height:200px; border-radius:12px; border:1px dashed #475569; display:flex; align-items:center; justify-content:center; color:#94a3b8;'>Gráfico dinámico de la convergencia (Plotly)</div>", unsafe_with_html=True)

# ==========================================
# PESTAÑA 2: SISTEMAS DE ECUACIONES
# ==========================================
with tab2:
    st.markdown("<br>", unsafe_with_html=True)
    metodo_sistemas = st.selectbox(
        "Seleccione el Método Matricial:",
        ["Eliminación de Gauss (Pivoteo Parcial)", "Gauss-Seidel"]
    )
    
    st.markdown("<div style='background-color:#1e293b; padding:25px; border-radius:12px; border:1px solid #334155;'>", unsafe_with_html=True)
    dimension = st.number_input("Dimensión del sistema (n x n):", min_value=2, max_value=10, value=3)
    st.info("Aquí el Desarrollador 3 generará las cajas de texto dinámicas para armar la matriz.")
    st.markdown("</div>", unsafe_with_html=True)

# ==========================================
# PESTAÑA 3: INTERPOLACIÓN E INTEGRACIÓN
# ==========================================
with tab3:
    st.markdown("<br>", unsafe_with_html=True)
    metodo_interp = st.selectbox(
        "Seleccione el Algoritmo:",
        ["Interpolación de Newton", "Interpolación de Lagrange", "Regla del Trapecio", "Reglas de Simpson"]
    )
    
    st.markdown("<div style='background-color:#1e293b; padding:25px; border-radius:12px; border:1px solid #334155;'>", unsafe_with_html=True)
    st.info("Aquí el Desarrollador 4 configurará la tabla para ingresar los vectores de puntos X e Y.")
    st.markdown("</div>", unsafe_with_html=True)
