import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


def graficar_biseccion(f, a, b, iterados, raiz, tolerancia, max_iter):
    """
    Genera gráfica interactiva para el método de Bisección usando Plotly.
    """
    x_vals = np.linspace(a - 0.5, b + 0.5, 400)
    y_vals = f(x_vals)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    n_iter = len(iterados)
    paleta = px.colors.sequential.Viridis
    n_colores = len(paleta)
    
    for i, (x, fx) in enumerate(iterados):
        idx = int(i / max(1, n_iter-1) * (n_colores - 1))
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[fx],
            mode='markers',
            name=f'Iteración {i+1}',
            marker=dict(
                size=10,
                color=paleta[idx],
                symbol='circle'
            ),
            hovertemplate=f'<b>Iteración {i+1}</b><br>x = {x:.6f}<br>f(x) = {fx:.6e}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[0, fx],
            mode='lines',
            showlegend=False,
            line=dict(dash='dash', color='gray', width=1),
            opacity=0.3
        ))
    
    if raiz is not None:
        fig.add_trace(go.Scatter(
            x=[raiz],
            y=[f(raiz)],
            mode='markers',
            name=f'Raíz ≈ {raiz:.8f}',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            hovertemplate=f'<b>Raíz</b><br>x = {raiz:.8f}<br>f(x) = {f(raiz):.6e}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[raiz, raiz],
            y=[0, f(raiz)],
            mode='lines',
            showlegend=False,
            line=dict(dash='solid', color='red', width=1),
            opacity=0.5
        ))
    
    fig.add_vline(x=a, line_dash="dot", line_color="gray", opacity=0.7, annotation_text=f'a = {a:.4f}')
    fig.add_vline(x=b, line_dash="dot", line_color="gray", opacity=0.7, annotation_text=f'b = {b:.4f}')
    
    fig.update_layout(
        title=dict(
            text=f'MÉTODO DE BISECCIÓN<br>Tolerancia = {tolerancia}, Iteraciones = {n_iter}',
            font=dict(size=16, color='#1E3A8A')
        ),
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=900,
        height=600
    )
    
    return fig

def graficar_newton_secante(f, dominio, iterados, raiz, metodo, tolerancia, max_iter):
    """
    Genera gráfica interactiva para Newton o Secante usando Plotly.
    """
    x_min, x_max = dominio
    if raiz is not None:
        x_min = min(x_min, raiz - 0.5)
        x_max = max(x_max, raiz + 0.5)
    
    x_vals = np.linspace(x_min, x_max, 500)
    y_vals = f(x_vals)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    
    n_iter = len(iterados)
    paleta = px.colors.sequential.Plasma
    n_colores = len(paleta)
    
    for i, (x, fx) in enumerate(iterados):
        idx = int(i / max(1, n_iter-1) * (n_colores - 1))
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[fx],
            mode='markers',
            name=f'Iteración {i+1}',
            marker=dict(
                size=10,
                color=paleta[idx],
                symbol='circle'
            ),
            hovertemplate=f'<b>Iteración {i+1}</b><br>x = {x:.6f}<br>f(x) = {fx:.6e}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[0, fx],
            mode='lines',
            showlegend=False,
            line=dict(dash='dash', color='gray', width=1),
            opacity=0.3
        ))
    
    if raiz is not None:
        fig.add_trace(go.Scatter(
            x=[raiz],
            y=[f(raiz)],
            mode='markers',
            name=f'Raíz ≈ {raiz:.8f}',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            hovertemplate=f'<b>Raíz</b><br>x = {raiz:.8f}<br>f(x) = {f(raiz):.6e}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'MÉTODO DE {metodo.upper()}<br>Tolerancia = {tolerancia}, Iteraciones = {n_iter}',
            font=dict(size=16, color='#1E3A8A')
        ),
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        width=900,
        height=600
    )
    
    return fig

def graficar_sistema_2d(A, b, solucion, metodo="Gauss"):
    """
    Grafica sistema 2x2 interactivo con Plotly.
    """
    # Extraer coeficientes
    a1, b1, c1 = A[0, 0], A[0, 1], b[0]
    a2, b2, c2 = A[1, 0], A[1, 1], b[1]
    
    # Generar puntos para las rectas
    x_vals = np.linspace(-10, 10, 400)
    
    fig = go.Figure()
    
    # Ecuación 1
    if abs(b1) > 1e-12:
        y1 = (c1 - a1 * x_vals) / b1
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y1,
            mode='lines',
            name=f'Ec. 1: {a1:.2f}x + {b1:.2f}y = {c1:.2f}',
            line=dict(color='blue', width=3)
        ))
    else:
        x_const = c1 / a1 if abs(a1) > 1e-12 else 0
        fig.add_vline(x=x_const, line_color='blue', annotation_text=f'x = {x_const:.2f}')
    
    # Ecuación 2
    if abs(b2) > 1e-12:
        y2 = (c2 - a2 * x_vals) / b2
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y2,
            mode='lines',
            name=f'Ec. 2: {a2:.2f}x + {b2:.2f}y = {c2:.2f}',
            line=dict(color='red', width=3)
        ))
    else:
        x_const = c2 / a2 if abs(a2) > 1e-12 else 0
        fig.add_vline(x=x_const, line_color='red', annotation_text=f'x = {x_const:.2f}')
    
    # Solución
    if solucion is not None and len(solucion) >= 2:
        fig.add_trace(go.Scatter(
            x=[solucion[0]],
            y=[solucion[1]],
            mode='markers',
            name=f'Solución: ({solucion[0]:.4f}, {solucion[1]:.4f})',
            marker=dict(
                size=20,
                color='green',
                symbol='star',
                line=dict(width=2, color='darkgreen')
            ),
            hovertemplate=f'<b>Solución</b><br>x = {solucion[0]:.4f}<br>y = {solucion[1]:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'SISTEMA LINEAL 2x2 - {metodo}',
        xaxis_title='x',
        yaxis_title='y',
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=600,
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def graficar_sistema_3d(A, b, solucion, metodo="Gauss"):
    """
    Grafica sistema 3x3 interactivo con Plotly (planos 3D rotables).
    """
    # Crear malla
    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)
    X, Y = np.meshgrid(x, y)
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    opacity = 0.6
    
    # Graficar cada plano
    for i in range(3):
        a1, a2, a3, c = A[i, 0], A[i, 1], A[i, 2], b[i]
        if abs(a3) > 1e-12:
            Z = (c - a1 * X - a2 * Y) / a3
            
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                name=f'Plano {i+1}',
                colorscale=[[0, colors[i]], [1, colors[i]]],
                showscale=False,
                opacity=opacity,
                hovertemplate=f'<b>Plano {i+1}</b><br>x = %{{x:.2f}}<br>y = %{{y:.2f}}<br>z = %{{z:.2f}}<extra></extra>'
            ))
    
    # Solución
    if solucion is not None and len(solucion) >= 3:
        fig.add_trace(go.Scatter3d(
            x=[solucion[0]],
            y=[solucion[1]],
            z=[solucion[2]],
            mode='markers',
            name=f'Solución: ({solucion[0]:.4f}, {solucion[1]:.4f}, {solucion[2]:.4f})',
            marker=dict(
                size=15,
                color='yellow',
                symbol='diamond',
                line=dict(width=2, color='black')
            ),
            hovertemplate=f'<b>Solución</b><br>x = {solucion[0]:.4f}<br>y = {solucion[1]:.4f}<br>z = {solucion[2]:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'SISTEMA LINEAL 3x3 - {metodo}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            zaxis=dict(range=[-10, 10])
        ),
        template='plotly_white',
        width=900,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def graficar_sistema_lineal(A, b, solucion, metodo="Gauss", n_dim=2):
    """
    Función principal que decide qué gráfica usar según la dimensión.
    """
    if n_dim == 2:
        return graficar_sistema_2d(A, b, solucion, metodo)
    elif n_dim == 3:
        return graficar_sistema_3d(A, b, solucion, metodo)
    else:
        return graficar_matriz_calor(A, b, solucion, metodo)


def graficar_matriz_calor(A, b, solucion, metodo="Gauss"):
    """
    Mapa de calor interactivo de la matriz A.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Matriz A', 'Vector Solución'),
        specs=[[{'type': 'heatmap'}, {'type': 'bar'}]]
    )
    
    # Mapa de calor de la matriz
    fig.add_trace(
        go.Heatmap(
            z=A,
            colorscale='RdBu_r',
            showscale=True,
            zmin=-np.max(np.abs(A)),
            zmax=np.max(np.abs(A)),
            text=np.round(A, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='Fila: %{y}<br>Columna: %{x}<br>Valor: %{z:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Barras del vector solución
    if solucion is not None:
        fig.add_trace(
            go.Bar(
                x=[f'x{i+1}' for i in range(len(solucion))],
                y=solucion,
                marker_color='#4ECDC4',
                text=np.round(solucion, 4),
                textposition='outside',
                hovertemplate='Variable: %{x}<br>Valor: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f'SISTEMA LINEAL {A.shape[0]}x{A.shape[1]} - {metodo}',
        template='plotly_white',
        width=900,
        height=500,
        showlegend=False
    )
    
    return fig


def graficar_interpolacion(x_datos, y_datos, polinomio, grado, titulo="Interpolación"):
    """
    Gráfica interactiva de interpolación con Plotly.
    """
    fig = go.Figure()
    
    # Puntos de datos
    fig.add_trace(go.Scatter(
        x=x_datos,
        y=y_datos,
        mode='markers',
        name='Datos',
        marker=dict(
            size=15,
            color='red',
            symbol='circle',
            line=dict(width=2, color='black')
        ),
        hovertemplate='<b>Dato</b><br>x = %{x:.4f}<br>y = %{y:.4f}<extra></extra>'
    ))
    
    # Polinomio interpolador
    x_interp = np.linspace(min(x_datos)-0.5, max(x_datos)+0.5, 300)
    y_interp = polinomio(x_interp)
    
    fig.add_trace(go.Scatter(
        x=x_interp,
        y=y_interp,
        mode='lines',
        name=f'Polinomio Grado {grado}',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title=f'{titulo} - Grado {grado}',
        xaxis_title='x',
        yaxis_title='y',
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def graficar_integracion(f, a, b, n, metodo, resultado):
    """
    Gráfica interactiva del área bajo la curva con Plotly.
    """
    # Generar puntos para la función
    x_vals = np.linspace(a - 0.5, b + 0.5, 500)
    y_vals = f(x_vals)
    
    # Puntos para el área bajo la curva
    x_fill = np.linspace(a, b, 200)
    y_fill = f(x_fill)
    
    fig = go.Figure()
    
    # Función
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='f(x)',
        line=dict(color='blue', width=3)
    ))
    
    # Área bajo la curva (relleno)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
        fill='toself',
        name=f'Área ≈ {resultado:.6f}',
        fillcolor='rgba(173, 216, 230, 0.5)',
        line=dict(color='rgba(173, 216, 230, 0)'),
        hovertemplate='Área: %{customdata:.6f}<extra></extra>'
    ))
    
    # Límites de integración
    fig.add_vline(x=a, line_dash="dot", line_color="gray", annotation_text=f'a = {a:.2f}')
    fig.add_vline(x=b, line_dash="dot", line_color="gray", annotation_text=f'b = {b:.2f}')
    
    fig.update_layout(
        title=f'INTEGRACIÓN NUMÉRICA - {metodo}<br>n = {n} segmentos, Resultado = {resultado:.8f}',
        xaxis_title='x',
        yaxis_title='f(x)',
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def graficar_convergencia_seidel(historial, metodo="Gauss-Seidel"):
    """
    Gráfica interactiva de la convergencia de Gauss-Seidel.
    """
    if not historial:
        return None
    
    historial = np.array(historial)
    n_iter = historial.shape[0]
    n_vars = historial.shape[1]
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i in range(n_vars):
        fig.add_trace(go.Scatter(
            x=list(range(1, n_iter + 1)),
            y=historial[:, i],
            mode='lines+markers',
            name=f'x{i+1}',
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=f'<b>x{i+1}</b><br>Iteración: %{{x}}<br>Valor: %{{y:.6f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Evolución de Variables - {metodo}',
        xaxis_title='Iteración',
        yaxis_title='Valor de la variable',
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig

def graficar_comparacion_errores(errores, iteraciones, metodo, titulo="Evolución del Error"):
    """
    Grafica la evolución del error en las iteraciones usando Plotly.
    
    Parámetros:
    - errores: lista de errores (float)
    - iteraciones: lista de números de iteración (int)
    - metodo: string con el nombre del método
    - titulo: string para el título de la gráfica
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iteraciones,
        y=errores,
        mode='lines+markers',
        name='Error',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=10, color='#4ECDC4', symbol='circle'),
        hovertemplate='<b>Iteración %{x}</b><br>Error: %{y:.6e}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{titulo} - {metodo}',
            font=dict(size=16, color='#1E3A8A')
        ),
        xaxis_title='Iteración',
        yaxis_title='Error',
        yaxis_type='log',  # Escala logarítmica para mejor visualización
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    return fig


def graficar_comparativa_sistemas(resultados):
    """
    Grafica comparativa de soluciones de diferentes métodos usando Plotly.
    
    Parámetros:
    - resultados: dict con estructura {metodo: {'solucion': array, 'iteraciones': int}}
    """
    if not resultados:
        return None
    
    metodos = list(resultados.keys())
    n_vars = len(resultados[metodos[0]]['solucion'])
    
    fig = go.Figure()
    
    # Colores para cada método
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA94D', '#A29BFE']
    
    for i, metodo in enumerate(metodos):
        sol = resultados[metodo]['solucion']
        
        fig.add_trace(go.Bar(
            x=[f'x{j+1}' for j in range(n_vars)],
            y=sol,
            name=metodo,
            marker_color=colors[i % len(colors)],
            text=[f'{val:.4f}' for val in sol],
            textposition='outside',
            hovertemplate=f'<b>{metodo}</b><br>Variable: %{{x}}<br>Valor: %{{y:.4f}}<extra></extra>'
        ))
    
    # Agregar información de iteraciones como anotaciones
    annotations = []
    for i, metodo in enumerate(metodos):
        iter_info = resultados[metodo].get('iteraciones', 'N/A')
        annotations.append(
            dict(
                x=0.15 + i * 0.2,
                y=1.05,
                xref='paper',
                yref='paper',
                text=f'{metodo}: {iter_info} iter.',
                showarrow=False,
                font=dict(size=11, color=colors[i % len(colors)])
            )
        )
    
    fig.update_layout(
        title=dict(
            text='Comparativa de Soluciones entre Métodos',
            font=dict(size=16, color='#1E3A8A')
        ),
        xaxis_title='Variables',
        yaxis_title='Valor',
        hovermode='x unified',
        template='plotly_white',
        width=900,
        height=500,
        barmode='group',
        bargap=0.15,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        annotations=annotations
    )
    
    return fig