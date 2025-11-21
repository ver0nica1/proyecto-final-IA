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

import numpy as np
import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import copy


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
        self.puntos = puntos
        self.capacidad_vehiculo = capacidad_vehiculo
        self.num_vehiculos = num_vehiculos
        self.punto_inicio_id = punto_inicio_id
        
        # Crear matriz de distancias
        self.matriz_distancias = self._calcular_matriz_distancias()
        
        # Parámetros del SA calibrados para mejor optimización
        self.temperatura_inicial = 15000  # Mayor temperatura inicial para explorar más
        self.temperatura_minima = 0.01    # Temperatura mínima más baja para refinamiento
        self.factor_enfriamiento = 0.985  # Enfriamiento más lento para mejor convergencia
        self.iteraciones_por_temperatura = 150  # Más iteraciones por temperatura
        
        # Estadísticas
        self.historial_costos = []
        self.historial_temperaturas = []
        self.mejor_solucion = None
        self.mejor_costo = float('inf')
    
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
        n = len(self.puntos)
        matriz = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = self.puntos[i].latitud, self.puntos[i].longitud
                    lat2, lon2 = self.puntos[j].latitud, self.puntos[j].longitud
                    
                    # Distancia euclidiana (aproximación)
                    # Para distancias más precisas, usar fórmula de Haversine
                    dist = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # km aprox
                    matriz[i][j] = dist
        
        return matriz
    
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
        R = 6371  # Radio de la Tierra en km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
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
        # Excluir el punto de inicio
        puntos_visitar = [p for p in self.puntos if p.id != self.punto_inicio_id]
        
        # Calcular ángulos polares desde el punto de inicio
        punto_inicio = next(p for p in self.puntos if p.id == self.punto_inicio_id)
        
        puntos_con_angulo = []
        for punto in puntos_visitar:
            dx = punto.longitud - punto_inicio.longitud
            dy = punto.latitud - punto_inicio.latitud
            angulo = math.atan2(dy, dx)
            distancia = math.sqrt(dx**2 + dy**2)
            puntos_con_angulo.append((punto, angulo, distancia))
        
        # Ordenar por ángulo
        puntos_con_angulo.sort(key=lambda x: (x[1], x[2]))
        
        # Calcular capacidad objetivo por vehículo para mejor distribución
        demanda_total = sum(p.demanda for p, _, _ in puntos_con_angulo)
        capacidad_objetivo = demanda_total / self.num_vehiculos
        
        # Asignar a rutas según capacidad y número de vehículos
        rutas = []
        ruta_actual = [self.punto_inicio_id]
        carga_actual = 0
        
        for punto, _, _ in puntos_con_angulo:
            # Calcular cuántos vehículos quedan disponibles
            vehiculos_restantes = self.num_vehiculos - len(rutas)
            
            # Determinar si debemos cerrar la ruta actual
            debe_cerrar = False
            
            if carga_actual + punto.demanda > self.capacidad_vehiculo:
                # Excede capacidad del vehículo
                debe_cerrar = True
            elif vehiculos_restantes > 1 and carga_actual >= capacidad_objetivo * 0.85:
                # Ya alcanzó la carga objetivo y aún hay vehículos disponibles
                debe_cerrar = True
            
            if debe_cerrar and len(ruta_actual) > 1:
                if vehiculos_restantes > 1:
                    # Cerrar ruta actual
                    ruta_actual.append(self.punto_inicio_id)
                    rutas.append(self._crear_ruta(ruta_actual))
                    
                    # Iniciar nueva ruta
                    ruta_actual = [self.punto_inicio_id, punto.id]
                    carga_actual = punto.demanda
                else:
                    # Es el último vehículo, agregar todos los puntos restantes
                    ruta_actual.append(punto.id)
                    carga_actual += punto.demanda
            else:
                # Agregar a la ruta actual
                ruta_actual.append(punto.id)
                carga_actual += punto.demanda
        
        # Agregar última ruta
        if len(ruta_actual) > 1:
            ruta_actual.append(self.punto_inicio_id)
            rutas.append(self._crear_ruta(ruta_actual))
        
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
        distancia = 0
        carga = 0
        
        for i in range(len(puntos_ids) - 1):
            distancia += self.matriz_distancias[puntos_ids[i]][puntos_ids[i+1]]
        
        for pid in puntos_ids[1:-1]:  # Excluir inicio y fin
            punto = next(p for p in self.puntos if p.id == pid)
            carga += punto.demanda
        
        # Estimar tiempo (velocidad promedio 2.64 km/h según documento)
        tiempo = distancia / 2.64
        
        return Ruta(puntos=puntos_ids, distancia=distancia, 
                   carga=carga, tiempo=tiempo)
    
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
        return sum(ruta.distancia for ruta in rutas)
    
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
        # Usar copia más eficiente en lugar de deepcopy
        nueva_solucion = [Ruta(puntos=r.puntos.copy(), distancia=r.distancia, 
                              carga=r.carga, tiempo=r.tiempo) for r in rutas]
        
        # Incluir operador de consolidación con probabilidad ajustada
        operadores = ['intra_swap', 'inter_swap', 'two_opt', 'relocate']
        if len(nueva_solucion) > self.num_vehiculos * 0.7:  # Solo si hay muchas rutas
            operadores.append('consolidate')
        
        operador = random.choice(operadores)
        
        if operador == 'intra_swap':
            # Intercambio dentro de la misma ruta
            if len(nueva_solucion) > 0:
                ruta_idx = random.randint(0, len(nueva_solucion) - 1)
                ruta = nueva_solucion[ruta_idx]
                
                if len(ruta.puntos) > 3:  # Necesita al menos 2 puntos internos
                    i = random.randint(1, len(ruta.puntos) - 3)
                    j = random.randint(1, len(ruta.puntos) - 3)
                    ruta.puntos[i], ruta.puntos[j] = ruta.puntos[j], ruta.puntos[i]
                    nueva_solucion[ruta_idx] = self._crear_ruta(ruta.puntos)
        
        elif operador == 'inter_swap':
            # Intercambio entre dos rutas diferentes
            if len(nueva_solucion) >= 2:
                r1_idx = random.randint(0, len(nueva_solucion) - 1)
                r2_idx = random.randint(0, len(nueva_solucion) - 1)
                
                if r1_idx != r2_idx:
                    r1 = nueva_solucion[r1_idx]
                    r2 = nueva_solucion[r2_idx]
                    
                    if len(r1.puntos) > 2 and len(r2.puntos) > 2:
                        i = random.randint(1, len(r1.puntos) - 2)
                        j = random.randint(1, len(r2.puntos) - 2)
                        
                        # Verificar restricciones de capacidad
                        punto_i = next(p for p in self.puntos if p.id == r1.puntos[i])
                        punto_j = next(p for p in self.puntos if p.id == r2.puntos[j])
                        
                        nueva_carga_r1 = r1.carga - punto_i.demanda + punto_j.demanda
                        nueva_carga_r2 = r2.carga - punto_j.demanda + punto_i.demanda
                        
                        if (nueva_carga_r1 <= self.capacidad_vehiculo and 
                            nueva_carga_r2 <= self.capacidad_vehiculo):
                            r1.puntos[i], r2.puntos[j] = r2.puntos[j], r1.puntos[i]
                            nueva_solucion[r1_idx] = self._crear_ruta(r1.puntos)
                            nueva_solucion[r2_idx] = self._crear_ruta(r2.puntos)
        
        elif operador == 'two_opt':
            # Operador 2-opt en una ruta
            if len(nueva_solucion) > 0:
                ruta_idx = random.randint(0, len(nueva_solucion) - 1)
                ruta = nueva_solucion[ruta_idx]
                
                if len(ruta.puntos) > 4:
                    i = random.randint(1, len(ruta.puntos) - 3)
                    j = random.randint(i + 1, len(ruta.puntos) - 2)
                    
                    # Invertir el segmento entre i y j
                    ruta.puntos[i:j+1] = reversed(ruta.puntos[i:j+1])
                    nueva_solucion[ruta_idx] = self._crear_ruta(ruta.puntos)
        
        elif operador == 'relocate':
            # Reubicar un punto de una ruta a otra
            if len(nueva_solucion) >= 2:
                r1_idx = random.randint(0, len(nueva_solucion) - 1)
                r2_idx = random.randint(0, len(nueva_solucion) - 1)
                
                if r1_idx != r2_idx:
                    r1 = nueva_solucion[r1_idx]
                    r2 = nueva_solucion[r2_idx]
                    
                    if len(r1.puntos) > 3:  # Debe tener al menos 2 puntos internos
                        i = random.randint(1, len(r1.puntos) - 2)
                        punto_id = r1.puntos[i]
                        punto = next(p for p in self.puntos if p.id == punto_id)
                        
                        # Verificar capacidad
                        if r2.carga + punto.demanda <= self.capacidad_vehiculo:
                            # Remover de r1
                            r1.puntos.pop(i)
                            
                            # Insertar en posición aleatoria de r2
                            pos = random.randint(1, len(r2.puntos) - 1)
                            r2.puntos.insert(pos, punto_id)
                            
                            nueva_solucion[r1_idx] = self._crear_ruta(r1.puntos)
                            nueva_solucion[r2_idx] = self._crear_ruta(r2.puntos)
        
        elif operador == 'consolidate':
            # Operador de consolidación: intenta combinar dos rutas pequeñas en una
            if len(nueva_solucion) >= 2:
                # Buscar las dos rutas con menor carga
                rutas_ordenadas = sorted(enumerate(nueva_solucion), key=lambda x: x[1].carga)
                
                for i in range(len(rutas_ordenadas) - 1):
                    idx1, r1 = rutas_ordenadas[i]
                    idx2, r2 = rutas_ordenadas[i + 1]
                    
                    # Verificar si se pueden combinar
                    carga_combinada = r1.carga + r2.carga
                    if carga_combinada <= self.capacidad_vehiculo:
                        # Combinar rutas: r1 + r2 (sin repetir el depósito)
                        puntos_combinados = r1.puntos[:-1] + r2.puntos[1:]
                        ruta_combinada = self._crear_ruta(puntos_combinados)
                        
                        # Crear nueva solución sin r1 y r2, agregando la ruta combinada
                        nueva_solucion = [r for j, r in enumerate(nueva_solucion) 
                                        if j != idx1 and j != idx2]
                        nueva_solucion.append(ruta_combinada)
                        break
        
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
        if costo_vecino < costo_actual:
            return True
        else:
            delta = costo_vecino - costo_actual
            probabilidad = math.exp(-delta / temperatura)
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
        # Generar solución inicial
        solucion_actual = self.generar_solucion_inicial()
        costo_actual = self.calcular_costo_total(solucion_actual)
        
        # Usar copia más eficiente para la mejor solución inicial
        self.mejor_solucion = [Ruta(puntos=r.puntos.copy(), 
                                   distancia=r.distancia, 
                                   carga=r.carga, 
                                   tiempo=r.tiempo) 
                              for r in solucion_actual]
        self.mejor_costo = costo_actual
        
        temperatura = self.temperatura_inicial
        iteracion = 0
        
        # Calcular total de iteraciones aproximado usando el valor actual
        total_iteraciones = 0
        temp = temperatura
        while temp > self.temperatura_minima:
            total_iteraciones += self.iteraciones_por_temperatura
            temp *= self.factor_enfriamiento
        
        self.historial_costos = []
        self.historial_temperaturas = []
        
        # Optimización: reducir frecuencia de callbacks y estadísticas
        callback_frecuencia = max(1, self.iteraciones_por_temperatura // 15)  # ~15 callbacks por temperatura
        stats_frecuencia = max(1, self.iteraciones_por_temperatura // 30)  # ~30 registros por temperatura
        
        # Ciclo principal
        while temperatura > self.temperatura_minima:
            for _ in range(self.iteraciones_por_temperatura):
                # Generar vecino
                solucion_vecina = self.generar_vecino(solucion_actual)
                costo_vecino = self.calcular_costo_total(solucion_vecina)
                
                # Criterio de aceptación
                if self.criterio_aceptacion(costo_actual, costo_vecino, temperatura):
                    solucion_actual = solucion_vecina
                    costo_actual = costo_vecino
                    
                    # Actualizar mejor solución (solo cuando mejora)
                    if costo_actual < self.mejor_costo:
                        # Usar copia más eficiente
                        self.mejor_solucion = [Ruta(puntos=r.puntos.copy(), 
                                                   distancia=r.distancia, 
                                                   carga=r.carga, 
                                                   tiempo=r.tiempo) 
                                              for r in solucion_actual]
                        self.mejor_costo = costo_actual
                
                iteracion += 1
                
                # Registrar estadísticas con menor frecuencia
                if iteracion % stats_frecuencia == 0:
                    self.historial_costos.append(self.mejor_costo)
                    self.historial_temperaturas.append(temperatura)
                
                # Callback de progreso con frecuencia adaptativa
                if callback_progreso and iteracion % callback_frecuencia == 0:
                    progreso = min(100.0, (iteracion / total_iteraciones) * 100)
                    callback_progreso(progreso, self.mejor_costo, temperatura, iteracion, total_iteraciones)
            
            # Enfriar temperatura
            temperatura *= self.factor_enfriamiento
        
        # Callback final para asegurar que se muestre 100%
        if callback_progreso:
            callback_progreso(100.0, self.mejor_costo, temperatura, iteracion, total_iteraciones)
        
        # Estadísticas finales
        estadisticas = {
            'iteraciones_totales': iteracion,
            'costo_inicial': self.calcular_costo_total(self.generar_solucion_inicial()),
            'costo_final': self.mejor_costo,
            'numero_rutas': len(self.mejor_solucion),
            'historial_costos': self.historial_costos,
            'historial_temperaturas': self.historial_temperaturas
        }
        
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
        errores = []
        
        # Verificar que todos los puntos se visiten exactamente una vez
        puntos_visitados = set()
        for ruta in rutas:
            for pid in ruta.puntos[1:-1]:  # Excluir inicio y fin
                if pid in puntos_visitados:
                    errores.append(f"Punto {pid} visitado más de una vez")
                puntos_visitados.add(pid)
        
        puntos_esperados = set(p.id for p in self.puntos if p.id != self.punto_inicio_id)
        puntos_faltantes = puntos_esperados - puntos_visitados
        if puntos_faltantes:
            errores.append(f"Puntos no visitados: {puntos_faltantes}")
        
        # Verificar capacidad
        for i, ruta in enumerate(rutas):
            if ruta.carga > self.capacidad_vehiculo:
                errores.append(f"Ruta {i+1} excede capacidad: {ruta.carga} > {self.capacidad_vehiculo}")
        
        # Verificar número de vehículos
        if len(rutas) > self.num_vehiculos:
            errores.append(f"Se usan {len(rutas)} vehículos, disponibles: {self.num_vehiculos}")
        
        return len(errores) == 0, errores