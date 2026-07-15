
from __future__ import annotations
import numpy as np



# 1. INTERPOLACIÓN DE NEWTON (Fig. 18.7)

def interpolacion_newton(x, y, xi):
    """
    Interpolación polinomial de Newton mediante diferencias divididas.

    Traducción directa de la SUBROUTINE NewtInt(x, y, n, xi, yint, ea):

        LOCAL fdd(n,n)
        DOFOR i = 0, n
            fdd(i,0) = y(i)
        DOFOR j = 1, n
            DOFOR i = 0, n-j
                fdd(i,j) = (fdd(i+1,j-1) - fdd(i,j-1)) / (x(i+j) - x(i))
        xterm = 1
        yint(0) = fdd(0,0)
        DOFOR order = 1, n
            xterm = xterm * (xi - x(order-1))
            yint2 = yint(order-1) + fdd(0,order) * xterm
            ea(order-1) = yint2 - yint(order-1)
            yint(order) = yint2

    Parámetros
    ----------
    x, y : array_like (n+1 puntos, x debe estar ordenado o no repetido)
    xi   : float o array_like, punto(s) donde se desea interpolar

    Retorna
    -------
    yint_final : float o ndarray
        Valor interpolado final (máximo orden posible, n = len(x)-1).
    yint_por_orden : ndarray (n+1, ) o (n+1, m)
        Predicción acumulada para cada orden 0..n (útil para elegir
        el grado óptimo, Ejemplo 18.5 del texto).
    ea : ndarray (n,) o (n, m)
        Error estimado (diferencia entre órdenes sucesivos) por cada
        incremento de orden.
    fdd : ndarray (n+1, n+1)
        Tabla de diferencias divididas (por si se desea inspeccionar).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x) - 1

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Los valores de x deben ser distintos entre sí.")

    xi_arr = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    m = xi_arr.size

    # --- Tabla de diferencias divididas (vectorizada por columnas) ---
    fdd = np.zeros((n + 1, n + 1), dtype=np.float64)
    fdd[:, 0] = y
    for j in range(1, n + 1):
        i = np.arange(0, n - j + 1)
        fdd[i, j] = (fdd[i + 1, j - 1] - fdd[i, j - 1]) / (x[i + j] - x[i])

    # --- Evaluación del polinomio (bucle secuencial: cada orden
    #     depende del acumulado anterior, no es paralelizable) ---
    yint_por_orden = np.zeros((n + 1, m), dtype=np.float64)
    ea = np.zeros((n, m), dtype=np.float64)

    xterm = np.ones(m, dtype=np.float64)
    yint_por_orden[0, :] = fdd[0, 0]

    for order in range(1, n + 1):
        xterm = xterm * (xi_arr - x[order - 1])
        yint2 = yint_por_orden[order - 1, :] + fdd[0, order] * xterm
        ea[order - 1, :] = yint2 - yint_por_orden[order - 1, :]
        yint_por_orden[order, :] = yint2

    yint_final = yint_por_orden[-1, :]

    if np.isscalar(xi) or (hasattr(xi, "ndim") and xi_arr.ndim == 0):
        return float(yint_final[0]), yint_por_orden[:, 0], ea[:, 0], fdd
    return yint_final, yint_por_orden, ea, fdd



# 2. INTERPOLACIÓN DE LAGRANGE (Fig. 18.11)

def interpolacion_lagrange(x, y, xi):
    """
    Interpolación polinomial de Lagrange, grado n (n+1 = número de datos).

    Traducción de la FUNCTION Lagrng(x, y, n, xi):

        sum = 0
        DOFOR i = 0, n
            product = y(i)
            DOFOR j = 0, n
                IF i != j THEN
                    product = product * (xi - x(j)) / (x(i) - x(j))
            sum = sum + product
        Lagrng = sum

    Parámetros
    ----------
    x, y : array_like (n+1 puntos)
    xi   : float o array_like

    Retorna
    -------
    float o ndarray con el valor interpolado.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError("x e y deben tener la misma longitud.")
    if len(np.unique(x)) != len(x):
        raise ValueError("Los valores de x deben ser distintos entre sí.")

    xi_arr = np.atleast_1d(np.asarray(xi, dtype=np.float64))
    n1 = len(x)

    # Para cada punto xi: term_ij = (xi - x_j)/(x_i - x_j) con i != j,
    # producto sobre j (vectorizado con máscara booleana), luego
    # multiplicado por y_i y sumado sobre i.
    resultado = np.zeros(xi_arr.size, dtype=np.float64)
    for i in range(n1):
        mask = np.ones(n1, dtype=bool)
        mask[i] = False
        # denom/numer: shape (n1-1,) para j != i, evaluado para todos los xi
        numer = xi_arr[:, None] - x[mask][None, :]         # (m, n1-1)
        denom = x[i] - x[mask]                               # (n1-1,)
        producto = np.prod(numer / denom, axis=1)           # (m,)
        resultado += y[i] * producto

    if np.isscalar(xi) or (hasattr(xi, "ndim") and xi_arr.ndim == 0):
        return float(resultado[0])
    return resultado



# 3. REGLA DEL TRAPECIO (Fig. 21.9 a y b)

def trapecio_simple(f0, f1, h):
    """
    Fig. 21.9 a):  FUNCTION Trap(h, f0, f1) = h * (f0 + f1) / 2
    """
    return h * (f0 + f1) / 2.0


def trapecio_compuesto(f, a, b, n):
    """
    Fig. 21.9 b):
        FUNCTION Trapm(h, n, f)
            sum = f0
            DOFOR i = 1, n-1
                sum = sum + 2*f_i
            sum = sum + f_n
            Trapm = h * sum / 2

    Parámetros
    ----------
    f : callable, función a integrar f(x)
    a, b : límites de integración
    n : número de segmentos (n >= 1)

    Retorna
    -------
    (integral, h) : valor de la integral y tamaño de paso usado.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    h = (b - a) / n
    x = a + h * np.arange(0, n + 1)          # n+1 puntos, vectorizado
    fx = np.asarray(f(x), dtype=np.float64)

    # sum = f0 + 2*sum(f_interiores) + f_n   (vectorizado, sin bucle)
    suma = fx[0] + fx[-1] + 2.0 * np.sum(fx[1:-1])
    integral = h * suma / 2.0
    return integral, h



# 4. REGLAS DE SIMPSON (Fig. 21.13 a, b, c, d)

def simpson_1_3(f0, f1, f2, h):
    """
    Fig. 21.13 a): Simp13(h, f0, f1, f2) = 2*h*(f0 + 4*f1 + f2) / 6
    """
    return 2.0 * h * (f0 + 4.0 * f1 + f2) / 6.0


def simpson_3_8(f0, f1, f2, f3, h):
    """
    Fig. 21.13 b): Simp38(h, f0, f1, f2, f3) = 3*h*(f0+3*(f1+f2)+f3) / 8
    """
    return 3.0 * h * (f0 + 3.0 * (f1 + f2) + f3) / 8.0


def simpson_1_3_compuesto(f, a, b, n):
    """
    Fig. 21.13 c): Simp13m(h, n, f)
        sum = f(x0)
        DOFOR i = 1, n-2, 2
            sum = sum + 4*f_i + 2*f_(i+1)
        sum = sum + 4*f_(n-1) + f_n
        Simp13m = h * sum / 3

    Requiere n par.
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("Simpson 1/3 compuesto requiere n par y >= 2.")
    h = (b - a) / n
    x = a + h * np.arange(0, n + 1)
    fx = np.asarray(f(x), dtype=np.float64)

    # Índices impares (1,3,5,...,n-1) llevan peso 4;
    # índices pares interiores (2,4,...,n-2) llevan peso 2 (vectorizado).
    impares = fx[1:-1:2]
    pares = fx[2:-1:2]
    suma = fx[0] + fx[-1] + 4.0 * np.sum(impares) + 2.0 * np.sum(pares)
    integral = h * suma / 3.0
    return integral, h


def simpson_integral(f, a, b, n):
    """
    Fig. 21.13 d): SimpInt(a, b, n, f) — integrador general que combina
    automáticamente Simpson 1/3 y 3/8 para manejar un número de
    segmentos par o impar:

        h = (b-a)/n
        IF n = 1 THEN
            sum = Trap(h, f0, fn)
        ELSE
            m = n
            odd = n/2 - INT(n/2)
            IF odd > 0 AND n > 1 THEN
                sum = Simp38(...) sobre los últimos 3 segmentos
                m = n - 3
            IF m > 1 THEN
                sum = sum + Simp13m(...) sobre los m segmentos restantes
        SimpInt = sum

    Parámetros
    ----------
    f : callable
    a, b : límites
    n : número de segmentos (n >= 1, puede ser par o impar)

    Retorna
    -------
    (integral, h)
    """
    if n < 1:
        raise ValueError("n debe ser >= 1.")
    h = (b - a) / n
    x = a + h * np.arange(0, n + 1)
    fx = np.asarray(f(x), dtype=np.float64)

    if n == 1:
        integral = trapecio_simple(fx[0], fx[1], h)
        return integral, h

    suma = 0.0
    m = n
    odd = (n % 2 != 0)

    if odd and n > 1:
        # Últimos 3 segmentos con Simpson 3/8
        suma += simpson_3_8(fx[n - 3], fx[n - 2], fx[n - 1], fx[n], h)
        m = n - 3

    if m > 1:
        # Segmentos restantes (pares) con Simpson 1/3 compuesto
        fx_rest = fx[0:m + 1]
        impares = fx_rest[1:-1:2]
        pares = fx_rest[2:-1:2]
        suma_13 = fx_rest[0] + fx_rest[-1] + 4.0 * np.sum(impares) + 2.0 * np.sum(pares)
        suma += h * suma_13 / 3.0
    elif m == 1:
        # Un único segmento remanente: trapecio
        suma += trapecio_simple(fx[0], fx[1], h)

    return suma, h



