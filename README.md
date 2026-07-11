# Sistema de Soluciones Numéricas — Equipo 5

Este proyecto consiste en una plataforma gráfica interactiva desarrollada en **Python** utilizando **Streamlit** y **Plotly**, diseñada para resolver, modelar y visualizar de forma analítica problemas de cálculo de raíces, sistemas de ecuaciones lineales, interpolación e integración numérica.

---

## Guía de Ejecución Paso a Paso (Para el Evaluador)

Siga estas instrucciones detalladas en su terminal de comandos para desplegar el sistema de forma local:

### Paso 1: Descargar y extraer el proyecto
*   Asegúrese de clonar este repositorio o descomprimir el archivo `.zip` con el código fuente en una carpeta local de su preferencia.

### Paso 2: Navegar a la raíz del directorio
*   Abra su terminal (Consola, PowerShell o CMD) y ubíquese dentro de la carpeta principal del proyecto ejecutando:
    ```bash
    cd "Ruta/De/La/Carpeta/Proyecto"
    ```

### Paso 3: Instalar los requerimientos del entorno
*   Instale las librerías del stack tecnológico ejecutando el siguiente comando (se requiere Python 3.9 o superior instalado):
    ```bash
    pip install -r requirements.txt
    ```
    *(Este paso configurará de forma automatizada `streamlit`, `plotly`, `numpy` y `sympy`)*.

### Paso 4: Levantar el servidor de la aplicación
*   Ejecute el comando para arrancar la interfaz web interactiva del sistema:
    
    streamlit run main.py

### Paso 5: Acceder al entorno gráfico
*   Una vez inicializado el servidor en la terminal, el sistema abrirá de manera automática una pestaña en su navegador de internet predeterminado. 
*   En caso de que no se abra automáticamente, copie y pegue la siguiente dirección local en la barra de su navegador:
    **`http://localhost:8501`**

---

## 🎨 Características de la Interfaz y Flujo de Trabajo

*   **Navegación por Pestañas:** Organización modular en una sola pantalla para conmutar rápidamente entre los tres bloques principales de la materia.
*   **Parámetros Globales:** Configuración unificada de criterios de parada (tolerancia de error e iteraciones máximas) desde la barra lateral.
*   **Arquitectura Desacoplada:** El archivo `main.py` actúa como el núcleo integrador. Captura las entradas del usuario desde la interfaz web, invoca las funciones lógicas alojadas en la carpeta `core/` pasándoles dichos parámetros, y despliega los resultados analíticos y gráficos devueltos por los algoritmos.

---

## 🛠️ Estructura del Repositorio

*   `main.py`: Archivo principal que gestiona la interfaz gráfica, los estilos visuales y la integración de los módulos.
*   `requirements.txt`: Archivo de configuración con las dependencias del sistema.
*   `core/`: Módulos lógicos independientes que contienen las implementaciones puras de los algoritmos matemáticos:
    *   `raices.py`: Métodos de Bisección, Newton-Raphson y la Secante.
    *   `sistemas_lineales.py`: Eliminación de Gauss y Gauss-Seidel.
    *   `interpolacion_integracion.py`: Esquemas de Newton, Lagrange, Trapecio y Simpson.
*   `utils/`:
    *   `graficador.py`: Lógica gráfica basada en Plotly para el trazado interactivo paso a paso de las curvas de convergencia.

---

## 👥 Integrantes del Equipo y Roles de Desarrollo
*   *Adrian Ortegano* (Desarrollador 1 - Integración, Arquitectura y UI)
*   *Ana * (Desarrollador 2 - Algoritmos de Cálculo de Raíces)
*   *Manuel Fuentes* (Desarrollador 3 - Algoritmos de Sistemas Lineales)
*   *Miguel Blanco* (Desarrollador 4 - Algoritmos de Interpolación e Integración)
*   *Carlos Flores* (Desarrollador 5 - Visualización Gráfica Interactiva con Plotly)
*   *Emilson Rivero* (Desarrollador 6 - Documentación, Casos de Estudio e Informe Técnico)