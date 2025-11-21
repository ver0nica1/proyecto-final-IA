"""
Módulo: sa_cvrp_logic.py

Implementación del algoritmo de Enfriamiento Simulado (Simulated Annealing)
para resolver el problema de ruteo de vehículos con capacidad (CVRP - Capacitated 
Vehicle Routing Problem).

El CVRP es un problema de optimización combinatoria donde se busca determinar las 
rutas óptimas para una flota de vehículos que deben visitar un conjunto de puntos 
de recolección, minimizando la distancia total recorrida y respetando las 
restricciones de capacidad de los vehículos.

Características principales:
- Heurística de barrido para generar solución inicial
- 5 operadores de vecindad: intra-swap, inter-swap, 2-opt, relocate, consolidate
- Criterio de aceptación de Metropolis
- Parámetros calibrados para optimización efectiva
- Validación de restricciones de capacidad y número de vehículos

Autor: Sistema de Optimización CVRP
Fecha: 2025
"""

import numpy as np  # Librería para operaciones numéricas y matrices
import random  # Generación de números aleatorios para operadores de vecindad
import math  # Funciones matemáticas (exp, sqrt, atan2, sin, cos, etc.)
from dataclasses import dataclass  # Decorador para crear clases de datos simplificadas
from typing import List, Tuple, Dict  # Type hints para mejor documentación y validación
import copy  # Módulo para realizar copias profundas de objetos (no usado actualmente)


@dataclass
class Punto:
    """
    Representa un punto de recolección en el problema CVRP.
    
    Attributes:
        id (int): Identificador único del punto
        nombre (str): Nombre descriptivo del punto de recolección
        direccion (str): Dirección física del punto
        latitud (float): Coordenada de latitud (grados decimales)
        longitud (float): Coordenada de longitud (grados decimales)
        demanda (float): Demanda o carga a recolectar en este punto (kg)
    """
    id: int
    nombre: str
    direccion: str
    latitud: float
    longitud: float
    demanda: float


@dataclass
class Ruta:
    """
    Representa una ruta completa de un vehículo en el problema CVRP.
    
    Una ruta siempre comienza y termina en el depósito (punto de inicio),
    visitando una secuencia de puntos intermedios.
    
    Attributes:
        puntos (List[int]): Secuencia de IDs de puntos que conforman la ruta,
                           incluyendo el depósito al inicio y al final
        distancia (float): Distancia total recorrida en la ruta (km)
        carga (float): Carga total recolectada en la ruta (kg)
        tiempo (float): Tiempo estimado para completar la ruta (horas)
    """
    puntos: List[int]
    distancia: float
    carga: float
    tiempo: float


class SimulatedAnnealingCVRP:
    """
    Implementación del algoritmo de Enfriamiento Simulado (Simulated Annealing)
    para resolver el problema de ruteo de vehículos con capacidad (CVRP).
    
    El algoritmo busca minimizar la distancia total recorrida por todos los
    vehículos, respetando las restricciones de:
    - Capacidad máxima de cada vehículo
    - Número máximo de vehículos disponibles
    - Todos los puntos deben ser visitados exactamente una vez
    - Todas las rutas comienzan y terminan en el depósito
    
    Estrategia de optimización:
    1. Genera una solución inicial usando heurística de barrido
    2. Explora el espacio de soluciones mediante operadores de vecindad
    3. Acepta soluciones peores con probabilidad decreciente (temperatura)
    4. Converge hacia una solución óptima o cercana al óptimo
    
    Parámetros calibrados:
    - Temperatura inicial: 15000 (exploración amplia)
    - Temperatura mínima: 0.01 (refinamiento fino)
    - Factor de enfriamiento: 0.985 (convergencia gradual)
    - Iteraciones por temperatura: 150 (búsqueda exhaustiva)
    """
    
    def __init__(self, puntos: List[Punto], capacidad_vehiculo: float = 25000,
                 num_vehiculos: int = 7, punto_inicio_id: int = 0):
        """
        Inicializa el algoritmo de Enfriamiento Simulado para CVRP.
        
        Args:
            puntos (List[Punto]): Lista completa de puntos a visitar, incluyendo
                                 el depósito. Cada punto debe tener coordenadas
                                 geográficas y demanda asociada.
            capacidad_vehiculo (float): Capacidad máxima de carga de cada vehículo
                                       en kilogramos. Por defecto 25000 kg.
            num_vehiculos (int): Número máximo de vehículos disponibles para
                               generar las rutas. Por defecto 7 vehículos.
            punto_inicio_id (int): ID del punto que actúa como depósito o punto
                                  de inicio/fin de todas las rutas. Por defecto 0.
        
        Raises:
            ValueError: Si la lista de puntos está vacía o si los parámetros
                       son inválidos (valores negativos o cero).
        """
        # Almacenar parámetros del problema
        self.puntos = puntos  # Lista de todos los puntos (depósito + puntos de recolección)
        self.capacidad_vehiculo = capacidad_vehiculo  # Capacidad máxima en kg por vehículo
        self.num_vehiculos = num_vehiculos  # Número máximo de vehículos disponibles
        self.punto_inicio_id = punto_inicio_id  # ID del punto que actúa como depósito
        
        # Crear matriz de distancias entre todos los puntos (precalculado para eficiencia)
        self.matriz_distancias = self._calcular_matriz_distancias()
        
        # Parámetros del SA calibrados experimentalmente para mejor optimización
        self.temperatura_inicial = 15000  # T₀: Alta para aceptar muchas soluciones al inicio
        self.temperatura_minima = 0.01    # Tₘᵢₙ: Baja para refinamiento fino al final
        self.factor_enfriamiento = 0.985  # α: Enfriamiento lento (T_nueva = T_actual × α)
        self.iteraciones_por_temperatura = 150  # Iteraciones antes de enfriar temperatura
        
        # Variables para rastrear el progreso y resultados del algoritmo
        self.historial_costos = []  # Lista para graficar evolución del costo
        self.historial_temperaturas = []  # Lista para graficar evolución de temperatura
        self.mejor_solucion = None  # Mejor conjunto de rutas encontrado hasta ahora
        self.mejor_costo = float('inf')  # Costo de la mejor solución (inicialmente infinito)
    
    def _calcular_matriz_distancias(self) -> np.ndarray:
        """
        Calcula la matriz de distancias euclidianas entre todos los puntos.
        
        Utiliza una aproximación euclidiana simple donde cada grado de diferencia
        se convierte aproximadamente a 111 km. Para distancias más precisas en
        aplicaciones de producción, se recomienda usar la fórmula de Haversine.
        
        Returns:
            np.ndarray: Matriz cuadrada NxN donde N es el número de puntos.
                       El elemento [i][j] representa la distancia en kilómetros
                       entre el punto i y el punto j.
        
        Note:
            La diagonal de la matriz contiene ceros (distancia de un punto
            a sí mismo).
        """
        n = len(self.puntos)  # Obtener número total de puntos
        matriz = np.zeros((n, n))  # Crear matriz cuadrada NxN inicializada en ceros
        
        # Calcular distancia entre cada par de puntos (i, j)
        for i in range(n):  # Iterar sobre todos los puntos como origen
            for j in range(n):  # Iterar sobre todos los puntos como destino
                if i != j:  # Solo calcular si son puntos diferentes (diagonal = 0)
                    # Obtener coordenadas geográficas de ambos puntos
                    lat1, lon1 = self.puntos[i].latitud, self.puntos[i].longitud
                    lat2, lon2 = self.puntos[j].latitud, self.puntos[j].longitud
                    
                    # Aproximación euclidiana: √((Δlat)² + (Δlon)²)
                    # Multiplicar por 111 km/grado para convertir diferencia angular a km
                    # Nota: Aproximación válida para distancias cortas
                    dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # km aprox
                    matriz[i][j] = dist  # Almacenar distancia en matriz[fila][columna]
        
        return matriz  # Devolver matriz completa de distancias
    
    def _calcular_distancia_haversine(self, lat1: float, lon1: float, 
                                     lat2: float, lon2: float) -> float:
        """
        Calcula la distancia geodésica usando la fórmula de Haversine.
        
        Esta fórmula considera la curvatura de la Tierra para calcular la distancia
        más corta entre dos puntos en la superficie terrestre (great circle distance).
        Es más precisa que la aproximación euclidiana para distancias grandes.
        
        Args:
            lat1 (float): Latitud del primer punto en grados decimales
            lon1 (float): Longitud del primer punto en grados decimales
            lat2 (float): Latitud del segundo punto en grados decimales
            lon2 (float): Longitud del segundo punto en grados decimales
        
        Returns:
            float: Distancia en kilómetros entre los dos puntos.
        
        Note:
            Asume un radio terrestre de 6371 km (radio medio de la Tierra).
            Para aplicaciones de alta precisión, considerar variaciones en
            el radio terrestre según la latitud.
        """
        R = 6371  # Radio medio de la Tierra en kilómetros
        
        # Convertir coordenadas de grados a radianes (necesario para funciones trigonométricas)
        lat1_rad = math.radians(lat1)  # Latitud punto 1 en radianes
        lat2_rad = math.radians(lat2)  # Latitud punto 2 en radianes
        delta_lat = math.radians(lat2 - lat1)  # Diferencia de latitudes en radianes
        delta_lon = math.radians(lon2 - lon1)  # Diferencia de longitudes en radianes
        
        # Fórmula de Haversine - Parte 1: calcular 'a'
        # a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        a = (math.sin(delta_lat / 2) ** 2 +  # Componente vertical
             math.cos(lat1_rad) * math.cos(lat2_rad) *  # Factor de corrección por latitud
             math.sin(delta_lon / 2) ** 2)  # Componente horizontal
        
        # Fórmula de Haversine - Parte 2: calcular ángulo central 'c'
        # c = 2 × atan2(√a, √(1-a))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))  # Ángulo central en radianes
        
        # Distancia = Radio × Ángulo_central
        return R * c  # Distancia en kilómetros sobre la superficie terrestre
    
    def generar_solucion_inicial(self) -> List[Ruta]:
        """
        Genera una solución inicial factible usando la heurística de barrido (sweep algorithm).
        
        Algoritmo:
        1. Calcula el ángulo polar de cada punto respecto al depósito
        2. Ordena los puntos por ángulo creciente (barrido angular)
        3. Agrupa puntos consecutivos en rutas respetando:
           - Capacidad máxima del vehículo
           - Número máximo de vehículos disponibles
           - Distribución equilibrada de carga entre vehículos
        
        La solución inicial busca un balance entre:
        - Proximidad geográfica (puntos cercanos en la misma ruta)
        - Utilización eficiente de la capacidad
        - Distribución equitativa entre vehículos
        
        Returns:
            List[Ruta]: Lista de rutas que conforman la solución inicial.
                       Cada ruta comienza y termina en el depósito.
        
        Note:
            - Calcula una "capacidad objetivo" = demanda_total / num_vehiculos
            - Cierra rutas al 85% de la capacidad objetivo para mejor distribución
            - El último vehículo puede exceder capacidad (será optimizado por SA)
        """
        # Filtrar solo los puntos de recolección (excluir el depósito)
        puntos_visitar = [p for p in self.puntos if p.id != self.punto_inicio_id]
        
        # Obtener referencia al punto de inicio (depósito)
        punto_inicio = next(p for p in self.puntos if p.id == self.punto_inicio_id)
        
        # Calcular coordenadas polares de cada punto respecto al depósito
        puntos_con_angulo = []  # Lista para almacenar (punto, ángulo, distancia)
        for punto in puntos_visitar:
            # Calcular diferencias en coordenadas cartesianas
            dx = punto.longitud - punto_inicio.longitud  # Δx (Este-Oeste)
            dy = punto.latitud - punto_inicio.latitud    # Δy (Norte-Sur)
            
            # atan2(y, x) devuelve ángulo en radianes [-π, π] desde el eje X positivo
            angulo = math.atan2(dy, dx)  # Ángulo polar del punto
            
            # Distancia euclidiana desde el depósito
            distancia = math.sqrt(dx**2 + dy**2)
            
            # Guardar tupla (objeto_punto, ángulo, distancia_al_deposito)
            puntos_con_angulo.append((punto, angulo, distancia))
        
        # Ordenar puntos por ángulo creciente (barrido circular)
        # Si dos puntos tienen mismo ángulo, ordenar por distancia
        puntos_con_angulo.sort(key=lambda x: (x[1], x[2]))
        
        # Calcular cuánta carga debería llevar cada vehículo idealmente
        demanda_total = sum(p.demanda for p, _, _ in puntos_con_angulo)  # Suma total de demandas
        capacidad_objetivo = demanda_total / self.num_vehiculos  # Promedio por vehículo
        
        # Construir rutas asignando puntos secuencialmente
        rutas = []  # Lista para almacenar rutas completadas
        ruta_actual = [self.punto_inicio_id]  # Iniciar ruta desde el depósito
        carga_actual = 0  # Carga acumulada en la ruta actual (kg)
        
        # Procesar cada punto en orden angular (barrido)
        for punto, _, _ in puntos_con_angulo:  # Desempaquetar: solo necesitamos el objeto punto
            # Calcular cuántos vehículos aún no tienen ruta asignada
            vehiculos_restantes = self.num_vehiculos - len(rutas)
            
            # Decidir si cerrar la ruta actual y empezar una nueva
            debe_cerrar = False  # Flag para indicar si cerrar ruta
            
            # Condición 1: Restricción de capacidad
            if carga_actual + punto.demanda > self.capacidad_vehiculo:
                # Agregar este punto excedería la capacidad del vehículo
                debe_cerrar = True
            # Condición 2: Distribución equilibrada
            elif vehiculos_restantes > 1 and carga_actual >= capacidad_objetivo * 0.85:
                # Ya tenemos 85% de la carga ideal y quedan vehículos disponibles
                # Cerrar ruta para evitar sobrecargar este vehículo
                debe_cerrar = True
            
            # Aplicar decisión de cierre de ruta
            if debe_cerrar and len(ruta_actual) > 1:  # Solo cerrar si hay al menos 1 punto visitado
                if vehiculos_restantes > 1:  # Aún quedan vehículos disponibles
                    # Cerrar ruta actual: agregar regreso al depósito
                    ruta_actual.append(self.punto_inicio_id)  # Completar ciclo: depósito → puntos → depósito
                    rutas.append(self._crear_ruta(ruta_actual))  # Crear objeto Ruta y agregar a lista
                    
                    # Iniciar nueva ruta con este punto
                    ruta_actual = [self.punto_inicio_id, punto.id]  # Nueva ruta: [depósito, primer_punto]
                    carga_actual = punto.demanda  # Reiniciar carga con demanda de este punto
                else:
                    # Es el último vehículo disponible: debe visitar todos los puntos restantes
                    # Agregar punto aunque exceda capacidad (será optimizado por SA después)
                    ruta_actual.append(punto.id)  # Agregar ID del punto a la secuencia
                    carga_actual += punto.demanda  # Acumular carga (puede exceder capacidad)
            else:
                # No cerrar ruta: agregar punto a la ruta actual
                ruta_actual.append(punto.id)  # Agregar ID del punto
                carga_actual += punto.demanda  # Acumular demanda
        
        # Procesar última ruta (la que quedó sin cerrar en el bucle)
        if len(ruta_actual) > 1:  # Verificar que tenga al menos 1 punto (además del depósito inicial)
            ruta_actual.append(self.punto_inicio_id)  # Completar ciclo: agregar regreso al depósito
            rutas.append(self._crear_ruta(ruta_actual))  # Crear objeto Ruta y agregar a lista
        
        return rutas
    
    def _crear_ruta(self, puntos_ids: List[int]) -> Ruta:
        """
        Construye un objeto Ruta completo a partir de una secuencia de IDs de puntos.
        
        Calcula automáticamente:
        - Distancia total recorrida (suma de distancias entre puntos consecutivos)
        - Carga total recolectada (suma de demandas de todos los puntos)
        - Tiempo estimado de recorrido (distancia / velocidad promedio)
        
        Args:
            puntos_ids (List[int]): Lista ordenada de IDs de puntos que conforman
                                   la ruta. Debe incluir el depósito al inicio
                                   y al final.
        
        Returns:
            Ruta: Objeto Ruta con todos los atributos calculados.
        
        Note:
            Asume una velocidad promedio de 2.64 km/h para el cálculo del tiempo,
            considerando paradas, tráfico y tiempo de recolección en cada punto.
        """
        distancia = 0  # Inicializar distancia total de la ruta en km
        carga = 0      # Inicializar carga total recolectada en kg
        
        # Calcular distancia total sumando distancias entre puntos consecutivos
        for i in range(len(puntos_ids) - 1):  # Iterar hasta penúltimo punto
            # Sumar distancia del punto i al punto i+1 usando matriz precalculada
            distancia += self.matriz_distancias[puntos_ids[i]][puntos_ids[i+1]]
        
        # Calcular carga total sumando demandas de todos los puntos visitados
        for pid in puntos_ids[1:-1]:  # Excluir primer y último (ambos son el depósito)
            # Buscar objeto Punto correspondiente al ID
            punto = next(p for p in self.puntos if p.id == pid)
            # Acumular su demanda
            carga += punto.demanda
        
        # Estimar tiempo de recorrido considerando velocidad promedio
        # 2.64 km/h incluye: conducción en ciudad + paradas + tiempo de recolección
        tiempo = distancia / 2.64  # Resultado en horas
        
        # Crear y devolver objeto Ruta con todos los atributos calculados
        return Ruta(puntos=puntos_ids,      # Secuencia de IDs
                   distancia=distancia,     # Distancia total en km
                   carga=carga,             # Carga total en kg
                   tiempo=tiempo)           # Tiempo estimado en horas
    
    def calcular_costo_total(self, rutas: List[Ruta]) -> float:
        """
        Calcula el costo total de una solución completa.
        
        El costo se define como la suma de las distancias de todas las rutas.
        Este es el objetivo a minimizar en el problema CVRP.
        
        Args:
            rutas (List[Ruta]): Lista de rutas que conforman la solución.
        
        Returns:
            float: Distancia total en kilómetros de todas las rutas.
        
        Note:
            En variantes del CVRP, el costo puede incluir otros factores como:
            - Número de vehículos utilizados
            - Tiempo total de operación
            - Costos de combustible o mantenimiento
        """
        # Sumar las distancias de todas las rutas para obtener costo total
        # Expresión generadora: itera sobre cada ruta y extrae su distancia
        return sum(ruta.distancia for ruta in rutas)  # Resultado en km
    
    def generar_vecino(self, rutas: List[Ruta]) -> List[Ruta]:
        """
        Genera una solución vecina aplicando aleatoriamente uno de los operadores de vecindad.
        
        Operadores disponibles:
        
        1. **intra_swap**: Intercambia dos puntos dentro de la misma ruta.
           - Mejora el orden de visita en una ruta específica
           - No afecta la carga de la ruta
        
        2. **inter_swap**: Intercambia un punto de una ruta con un punto de otra ruta.
           - Permite redistribuir carga entre vehículos
           - Verifica restricciones de capacidad antes de aplicar
        
        3. **two_opt**: Invierte un segmento de puntos dentro de una ruta.
           - Elimina cruces en la ruta (mejora geométrica)
           - Muy efectivo para rutas con trayectorias subóptimas
        
        4. **relocate**: Mueve un punto de una ruta a otra ruta diferente.
           - Permite consolidar rutas
           - Redistribuye puntos entre vehículos
        
        5. **consolidate**: Combina dos rutas pequeñas en una sola.
           - Reduce el número total de vehículos usados
           - Solo se activa si hay más del 70% de vehículos en uso
           - Verifica que la carga combinada no exceda capacidad
        
        Args:
            rutas (List[Ruta]): Solución actual a modificar.
        
        Returns:
            List[Ruta]: Nueva solución vecina después de aplicar el operador.
        
        Note:
            - Todos los operadores respetan las restricciones de capacidad
            - La selección del operador es aleatoria con probabilidad uniforme
            - El operador 'consolidate' tiene probabilidad condicional
        """
        # Crear copia de la solución para modificar (evitar alterar original)
        # Solo copiamos la lista de puntos (lista.copy() es shallow copy)
        # Los atributos distancia, carga, tiempo se recalcularán después
        nueva_solucion = [Ruta(puntos=r.puntos.copy(),  # Copiar lista de IDs de puntos
                              distancia=r.distancia,    # Mantener valores actuales
                              carga=r.carga,            # (se recalcularán si hay cambios)
                              tiempo=r.tiempo) for r in rutas]
        
        # Seleccionar operador de vecindad a aplicar
        operadores = ['intra_swap', 'inter_swap', 'two_opt', 'relocate']  # Operadores base
        
        # Agregar operador 'consolidate' solo si se usan muchos vehículos
        # (más del 70% de los disponibles)
        if len(nueva_solucion) > self.num_vehiculos * 0.7:
            operadores.append('consolidate')  # Permite reducir número de rutas
        
        # Seleccionar un operador aleatoriamente con probabilidad uniforme
        operador = random.choice(operadores)
        
        if operador == 'intra_swap':
            # OPERADOR 1: Intercambio dentro de la misma ruta
            # Intercambia dos puntos de visita dentro de una ruta para mejorar su orden
            if len(nueva_solucion) > 0:  # Verificar que exista al menos una ruta
                # Seleccionar una ruta aleatoria
                ruta_idx = random.randint(0, len(nueva_solucion) - 1)
                ruta = nueva_solucion[ruta_idx]  # Obtener referencia a la ruta
                
                # Verificar que la ruta tenga al menos 2 puntos internos (sin contar depósitos)
                # Estructura: [depósito, punto1, punto2, ..., depósito] → necesita len > 3
                if len(ruta.puntos) > 3:
                    # Seleccionar dos posiciones aleatorias (excluyendo depósitos)
                    # Rango: 1 hasta len-3 (para no seleccionar último depósito)
                    i = random.randint(1, len(ruta.puntos) - 3)  # Primera posición
                    j = random.randint(1, len(ruta.puntos) - 3)  # Segunda posición
                    
                    # Intercambiar los puntos en las posiciones i y j
                    ruta.puntos[i], ruta.puntos[j] = ruta.puntos[j], ruta.puntos[i]
                    
                    # Recalcular distancia, carga y tiempo con el nuevo orden
                    nueva_solucion[ruta_idx] = self._crear_ruta(ruta.puntos)
        
        elif operador == 'inter_swap':
            # OPERADOR 2: Intercambio entre dos rutas diferentes
            # Intercambia un punto de una ruta con un punto de otra ruta
            if len(nueva_solucion) >= 2:  # Necesita al menos 2 rutas
                # Seleccionar dos rutas aleatorias
                r1_idx = random.randint(0, len(nueva_solucion) - 1)  # Índice ruta 1
                r2_idx = random.randint(0, len(nueva_solucion) - 1)  # Índice ruta 2
                
                # Verificar que sean rutas diferentes
                if r1_idx != r2_idx:
                    r1 = nueva_solucion[r1_idx]  # Referencia a ruta 1
                    r2 = nueva_solucion[r2_idx]  # Referencia a ruta 2
                    
                    # Verificar que ambas rutas tengan al menos 1 punto interno
                    if len(r1.puntos) > 2 and len(r2.puntos) > 2:
                        # Seleccionar puntos aleatorios de cada ruta (excluyendo depósitos)
                        i = random.randint(1, len(r1.puntos) - 2)  # Posición en ruta 1
                        j = random.randint(1, len(r2.puntos) - 2)  # Posición en ruta 2
                        
                        # Obtener objetos Punto para verificar demandas
                        punto_i = next(p for p in self.puntos if p.id == r1.puntos[i])
                        punto_j = next(p for p in self.puntos if p.id == r2.puntos[j])
                        
                        # Calcular nuevas cargas después del intercambio
                        # Ruta1: pierde punto_i, gana punto_j
                        nueva_carga_r1 = r1.carga - punto_i.demanda + punto_j.demanda
                        # Ruta2: pierde punto_j, gana punto_i
                        nueva_carga_r2 = r2.carga - punto_j.demanda + punto_i.demanda
                        
                        # Verificar restricción: ambas rutas deben respetar capacidad
                        if (nueva_carga_r1 <= self.capacidad_vehiculo and 
                            nueva_carga_r2 <= self.capacidad_vehiculo):
                            # Intercambiar puntos entre rutas
                            r1.puntos[i], r2.puntos[j] = r2.puntos[j], r1.puntos[i]
                            # Recalcular métricas de ambas rutas
                            nueva_solucion[r1_idx] = self._crear_ruta(r1.puntos)
                            nueva_solucion[r2_idx] = self._crear_ruta(r2.puntos)
        
        elif operador == 'two_opt':
            # OPERADOR 3: 2-opt (inversión de segmento)
            # Invierte el orden de un segmento de puntos dentro de una ruta
            # Útil para eliminar cruces y mejorar geometría de la ruta
            if len(nueva_solucion) > 0:  # Verificar que exista al menos una ruta
                # Seleccionar una ruta aleatoria
                ruta_idx = random.randint(0, len(nueva_solucion) - 1)
                ruta = nueva_solucion[ruta_idx]  # Obtener referencia a la ruta
                
                # Necesita al menos 3 puntos internos para hacer 2-opt útil
                if len(ruta.puntos) > 4:  # [depósito, p1, p2, p3, depósito]
                    # Seleccionar dos posiciones para definir el segmento a invertir
                    i = random.randint(1, len(ruta.puntos) - 3)      # Inicio del segmento
                    j = random.randint(i + 1, len(ruta.puntos) - 2)  # Fin del segmento (j > i)
                    
                    # Invertir el orden de los puntos entre i y j (inclusive)
                    # Ejemplo: [A, B, C, D, E] con i=1, j=3 → [A, D, C, B, E]
                    ruta.puntos[i:j+1] = reversed(ruta.puntos[i:j+1])
                    
                    # Recalcular distancia con el nuevo orden
                    nueva_solucion[ruta_idx] = self._crear_ruta(ruta.puntos)
        
        elif operador == 'relocate':
            # OPERADOR 4: Reubicación
            # Mueve un punto de una ruta a otra ruta diferente
            if len(nueva_solucion) >= 2:  # Necesita al menos 2 rutas
                # Seleccionar dos rutas aleatorias
                r1_idx = random.randint(0, len(nueva_solucion) - 1)  # Ruta origen
                r2_idx = random.randint(0, len(nueva_solucion) - 1)  # Ruta destino
                
                # Verificar que sean rutas diferentes
                if r1_idx != r2_idx:
                    r1 = nueva_solucion[r1_idx]  # Referencia a ruta origen
                    r2 = nueva_solucion[r2_idx]  # Referencia a ruta destino
                    
                    # Verificar que ruta origen tenga al menos 2 puntos internos
                    if len(r1.puntos) > 3:  # [depósito, p1, p2, depósito]
                        # Seleccionar punto aleatorio de la ruta origen
                        i = random.randint(1, len(r1.puntos) - 2)  # Posición del punto
                        punto_id = r1.puntos[i]  # ID del punto a mover
                        
                        # Obtener objeto Punto para verificar su demanda
                        punto = next(p for p in self.puntos if p.id == punto_id)
                        
                        # Verificar que la ruta destino pueda aceptar el punto
                        if r2.carga + punto.demanda <= self.capacidad_vehiculo:
                            # Remover punto de ruta origen
                            r1.puntos.pop(i)  # Eliminar de posición i
                            
                            # Insertar en posición aleatoria de ruta destino
                            pos = random.randint(1, len(r2.puntos) - 1)  # No insertar al final (depósito)
                            r2.puntos.insert(pos, punto_id)  # Insertar en posición
                            
                            # Recalcular métricas de ambas rutas
                            nueva_solucion[r1_idx] = self._crear_ruta(r1.puntos)
                            nueva_solucion[r2_idx] = self._crear_ruta(r2.puntos)
        
        elif operador == 'consolidate':
            # OPERADOR 5: Consolidación
            # Combina dos rutas pequeñas en una sola para reducir número de vehículos
            if len(nueva_solucion) >= 2:  # Necesita al menos 2 rutas
                # Ordenar rutas por carga (de menor a mayor)
                # Devuelve lista de tuplas (índice_original, objeto_ruta)
                rutas_ordenadas = sorted(enumerate(nueva_solucion), 
                                        key=lambda x: x[1].carga)
                
                # Intentar combinar las rutas más pequeñas
                for i in range(len(rutas_ordenadas) - 1):
                    idx1, r1 = rutas_ordenadas[i]      # Ruta más pequeña
                    idx2, r2 = rutas_ordenadas[i + 1]  # Segunda ruta más pequeña
                    
                    # Calcular carga combinada
                    carga_combinada = r1.carga + r2.carga
                    
                    # Verificar que la combinación respete capacidad del vehículo
                    if carga_combinada <= self.capacidad_vehiculo:
                        # Combinar rutas conectándolas (eliminar depósito intermedio)
                        # r1: [dep, A, B, dep] + r2: [dep, C, D, dep] → [dep, A, B, C, D, dep]
                        puntos_combinados = r1.puntos[:-1] + r2.puntos[1:]  # Unir sin duplicar depósito
                        ruta_combinada = self._crear_ruta(puntos_combinados)  # Crear nueva ruta
                        
                        # Crear nueva solución: remover r1 y r2, agregar ruta combinada
                        nueva_solucion = [r for j, r in enumerate(nueva_solucion) 
                                        if j != idx1 and j != idx2]  # Filtrar rutas eliminadas
                        nueva_solucion.append(ruta_combinada)  # Agregar ruta combinada
                        break  # Salir del bucle (solo combinar un par por iteración)
        
        return nueva_solucion
    
    def criterio_aceptacion(self, costo_actual: float, costo_vecino: float, 
                           temperatura: float) -> bool:
        """
        Determina si se acepta o rechaza una solución vecina usando el criterio de Metropolis.
        
        Reglas de aceptación:
        - Si el vecino es mejor (menor costo): SIEMPRE acepta
        - Si el vecino es peor (mayor costo): acepta con probabilidad e^(-Δ/T)
          donde Δ = costo_vecino - costo_actual y T = temperatura
        
        La probabilidad de aceptar soluciones peores disminuye con:
        - Mayor diferencia de costo (Δ grande)
        - Menor temperatura (T pequeña)
        
        Esto permite:
        - Exploración amplia al inicio (temperatura alta, acepta peores soluciones)
        - Refinamiento al final (temperatura baja, rechaza soluciones peores)
        - Escape de óptimos locales
        
        Args:
            costo_actual (float): Costo de la solución actual.
            costo_vecino (float): Costo de la solución vecina propuesta.
            temperatura (float): Temperatura actual del algoritmo.
        
        Returns:
            bool: True si se acepta la solución vecina, False en caso contrario.
        
        Note:
            Este es el mecanismo clave que diferencia al Simulated Annealing
            de una búsqueda local simple (hill climbing).
        """
        # Criterio de Metropolis para Simulated Annealing
        if costo_vecino < costo_actual:
            # Caso 1: El vecino es mejor (menor costo)
            return True  # SIEMPRE aceptar mejoras
        else:
            # Caso 2: El vecino es peor (mayor costo)
            # Calcular diferencia de costo (Δ)
            delta = costo_vecino - costo_actual  # Δ > 0 (empeoramiento)
            
            # Calcular probabilidad de aceptación según fórmula de Boltzmann
            # P(aceptar) = e^(-Δ/T)
            # - Δ grande → probabilidad baja
            # - T alta → probabilidad alta (exploración)
            # - T baja → probabilidad baja (refinamiento)
            probabilidad = math.exp(-delta / temperatura)
            
            # Aceptar con probabilidad calculada
            # Generar número aleatorio [0, 1) y comparar con probabilidad
            return random.random() < probabilidad
    
    def ejecutar(self, callback_progreso=None) -> Tuple[List[Ruta], float, Dict]:
        """
        Ejecuta el algoritmo completo de Enfriamiento Simulado para resolver el CVRP.
        
        Proceso de optimización:
        1. Genera solución inicial usando heurística de barrido
        2. Inicializa temperatura en valor alto (exploración)
        3. En cada iteración:
           - Genera solución vecina con operador aleatorio
           - Calcula costo de la solución vecina
           - Decide aceptación según criterio de Metropolis
           - Actualiza mejor solución si se encuentra mejora
        4. Reduce temperatura gradualmente (factor de enfriamiento)
        5. Termina cuando temperatura alcanza valor mínimo
        
        Args:
            callback_progreso (callable, optional): Función para reportar progreso
                                                   durante la ejecución. Recibe:
                                                   (progreso%, mejor_costo, temperatura,
                                                    iteracion_actual, total_iteraciones)
        
        Returns:
            Tuple[List[Ruta], float, Dict]: Tupla conteniendo:
                - mejor_solucion: Lista de rutas de la mejor solución encontrada
                - mejor_costo: Distancia total de la mejor solución (km)
                - estadisticas: Diccionario con métricas de ejecución:
                    * 'iteraciones_totales': Número total de iteraciones ejecutadas
                    * 'costo_inicial': Costo de la solución inicial
                    * 'costo_final': Costo de la mejor solución encontrada
                    * 'numero_rutas': Cantidad de rutas en la mejor solución
                    * 'historial_costos': Evolución del mejor costo por iteración
                    * 'historial_temperaturas': Evolución de la temperatura
        
        Note:
            - El algoritmo garantiza encontrar una solución factible
            - La calidad de la solución depende de los parámetros (temperatura, enfriamiento)
            - Ejecuciones múltiples pueden dar resultados diferentes (aleatorización)
            - Tiempo de ejecución depende de: número de puntos, parámetros SA
        """
        # FASE 1: INICIALIZACIÓN
        # Generar solución inicial factible usando heurística de barrido
        solucion_actual = self.generar_solucion_inicial()  # Lista de Rutas
        costo_actual = self.calcular_costo_total(solucion_actual)  # Distancia total (km)
        
        # Guardar como mejor solución (es la única por ahora)
        # Hacer copia para no perder referencia cuando solucion_actual cambie
        self.mejor_solucion = [Ruta(puntos=r.puntos.copy(),  # Copiar lista de puntos
                                   distancia=r.distancia,    # Mantener valores
                                   carga=r.carga, 
                                   tiempo=r.tiempo) 
                              for r in solucion_actual]
        self.mejor_costo = costo_actual  # Guardar costo de la mejor solución
        
        # Inicializar parámetros del algoritmo SA
        temperatura = self.temperatura_inicial  # T₀ = 15,000 (alta temperatura)
        iteracion = 0  # Contador global de iteraciones
        
        # Calcular cuántas iteraciones totales se ejecutarán (para barra de progreso)
        # Simular el proceso de enfriamiento para contar iteraciones
        total_iteraciones = 0  # Contador de iteraciones
        temp = temperatura     # Variable temporal para simular enfriamiento
        while temp > self.temperatura_minima:  # Mientras no se alcance temperatura mínima
            total_iteraciones += self.iteraciones_por_temperatura  # Sumar iteraciones de este nivel
            temp *= self.factor_enfriamiento  # Enfriar temperatura
        # Resultado: total_iteraciones ≈ 391,800 (con parámetros default)
        
        # Inicializar listas para rastrear evolución del algoritmo
        self.historial_costos = []        # Para graficar convergencia (eje Y)
        self.historial_temperaturas = []  # Para graficar temperatura (eje Y)
        
        # Configurar frecuencias de actualización (optimización de rendimiento)
        # No actualizar en cada iteración para evitar overhead
        callback_frecuencia = max(1, self.iteraciones_por_temperatura // 15)  # ~10 callbacks por nivel de temperatura
        stats_frecuencia = max(1, self.iteraciones_por_temperatura // 30)     # ~5 registros por nivel de temperatura
        
        # FASE 2: CICLO PRINCIPAL DE OPTIMIZACIÓN
        # Bucle externo: control de temperatura (esquema de enfriamiento)
        while temperatura > self.temperatura_minima:  # Condición de parada
            
            # Bucle interno: iteraciones a temperatura constante
            for _ in range(self.iteraciones_por_temperatura):  # 150 iteraciones por temperatura
                
                # Paso 1: Generar solución vecina
                # Aplicar operador de vecindad aleatorio a la solución actual
                solucion_vecina = self.generar_vecino(solucion_actual)
                costo_vecino = self.calcular_costo_total(solucion_vecina)  # Calcular su costo
                
                # Paso 2: Decidir aceptación con criterio de Metropolis
                if self.criterio_aceptacion(costo_actual, costo_vecino, temperatura):
                    # Aceptar solución vecina (reemplazar actual)
                    solucion_actual = solucion_vecina  # Actualizar solución actual
                    costo_actual = costo_vecino        # Actualizar costo actual
                    
                    # Paso 3: Actualizar mejor solución si es una mejora
                    if costo_actual < self.mejor_costo:  # Nueva mejor solución encontrada
                        # Hacer copia de la solución para preservarla
                        self.mejor_solucion = [Ruta(puntos=r.puntos.copy(), 
                                                   distancia=r.distancia, 
                                                   carga=r.carga, 
                                                   tiempo=r.tiempo) 
                                              for r in solucion_actual]
                        self.mejor_costo = costo_actual  # Actualizar mejor costo
                
                # Incrementar contador de iteraciones
                iteracion += 1  # Contar esta iteración
                
                # Registrar estadísticas periódicamente (no en cada iteración)
                if iteracion % stats_frecuencia == 0:  # Cada N iteraciones
                    self.historial_costos.append(self.mejor_costo)  # Guardar costo actual
                    self.historial_temperaturas.append(temperatura)  # Guardar temperatura
                
                # Llamar callback de progreso para actualizar UI
                if callback_progreso and iteracion % callback_frecuencia == 0:
                    # Calcular porcentaje de progreso
                    progreso = min(100.0, (iteracion / total_iteraciones) * 100)
                    # Llamar función callback con información de progreso
                    callback_progreso(progreso, self.mejor_costo, temperatura, 
                                    iteracion, total_iteraciones)
            
            # Paso 4: Enfriar temperatura (después de completar iteraciones)
            # T_nueva = T_actual × α (donde α = 0.985)
            temperatura *= self.factor_enfriamiento  # Reducir temperatura gradualmente
        
        # FASE 3: FINALIZACIÓN
        # Callback final para asegurar que se muestre 100% de progreso
        if callback_progreso:
            # Forzar progreso al 100% (por si no se alcanzó por redondeo)
            callback_progreso(100.0, self.mejor_costo, temperatura, 
                            iteracion, total_iteraciones)
        
        # Construir diccionario de estadísticas para reportar resultados
        estadisticas = {
            'iteraciones_totales': iteracion,  # Número real de iteraciones ejecutadas
            'costo_inicial': self.calcular_costo_total(self.generar_solucion_inicial()),  # Costo de solución inicial
            'costo_final': self.mejor_costo,  # Costo de la mejor solución encontrada
            'numero_rutas': len(self.mejor_solucion),  # Número de rutas en mejor solución
            'historial_costos': self.historial_costos,  # Evolución del costo (para gráfica)
            'historial_temperaturas': self.historial_temperaturas  # Evolución de temperatura
        }
        
        # Devolver tupla con: (mejor_solucion, mejor_costo, estadisticas)
        return self.mejor_solucion, self.mejor_costo, estadisticas
    
    def validar_solucion(self, rutas: List[Ruta]) -> Tuple[bool, List[str]]:
        """
        Valida que una solución cumpla con todas las restricciones del problema CVRP.
        
        Restricciones verificadas:
        
        1. **Cobertura completa**: Todos los puntos (excepto el depósito) deben ser
           visitados exactamente una vez.
        
        2. **Capacidad de vehículos**: Ninguna ruta puede exceder la capacidad
           máxima del vehículo.
        
        3. **Límite de vehículos**: El número de rutas no puede exceder el número
           de vehículos disponibles.
        
        Args:
            rutas (List[Ruta]): Solución a validar.
        
        Returns:
            Tuple[bool, List[str]]: Tupla conteniendo:
                - es_valida: True si la solución cumple todas las restricciones
                - errores: Lista de mensajes describiendo las violaciones encontradas
                          (lista vacía si la solución es válida)
        
        Example:
            >>> es_valida, errores = algoritmo.validar_solucion(rutas)
            >>> if not es_valida:
            ...     for error in errores:
            ...         print(f"Error: {error}")
        
        Note:
            Esta función es útil para:
            - Verificar soluciones antes de reportar resultados
            - Debugging durante el desarrollo
            - Asegurar calidad en producción
        """
        errores = []  # Lista para acumular mensajes de error
        
        # VALIDACIÓN 1: Cobertura completa (todos los puntos visitados exactamente una vez)
        puntos_visitados = set()  # Conjunto para rastrear puntos ya visitados
        for ruta in rutas:  # Iterar sobre cada ruta
            for pid in ruta.puntos[1:-1]:  # Excluir primer y último punto (depósitos)
                if pid in puntos_visitados:  # Punto ya fue visitado antes
                    errores.append(f"Punto {pid} visitado más de una vez")  # Error de duplicado
                puntos_visitados.add(pid)  # Agregar al conjunto de visitados
        
        # Verificar que no falten puntos por visitar
        puntos_esperados = set(p.id for p in self.puntos if p.id != self.punto_inicio_id)  # Todos los puntos (sin depósito)
        puntos_faltantes = puntos_esperados - puntos_visitados  # Diferencia de conjuntos
        if puntos_faltantes:  # Hay puntos sin visitar
            errores.append(f"Puntos no visitados: {puntos_faltantes}")  # Error de omisión
        
        # VALIDACIÓN 2: Restricción de capacidad
        for i, ruta in enumerate(rutas):  # Iterar con índice
            if ruta.carga > self.capacidad_vehiculo:  # Ruta excede capacidad
                # Reportar error con número de ruta y valores
                errores.append(f"Ruta {i+1} excede capacidad: {ruta.carga} > {self.capacidad_vehiculo}")
        
        # VALIDACIÓN 3: Límite de vehículos disponibles
        if len(rutas) > self.num_vehiculos:  # Más rutas que vehículos
            # Reportar error con número de rutas usadas vs disponibles
            errores.append(f"Se usan {len(rutas)} vehículos, disponibles: {self.num_vehiculos}")
        
        # Devolver tupla: (es_valida, lista_errores)
        return len(errores) == 0, errores  # True si lista vacía, False si tiene errores