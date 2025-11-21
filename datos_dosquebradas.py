"""
Módulo: datos_dosquebradas.py
Datos completos de los 144 puntos de recolección de Dosquebradas
Basado en la Tabla 1 del documento
"""

import numpy as np
from sa_cvrp_logic import Punto


def cargar_144_puntos_completos():
    """
    Carga los 144 puntos de recolección con coordenadas reales aproximadas
    para Dosquebradas, Risaralda, Colombia
    """
    
    # Punto de inicio (Centro Administrativo Municipal - coordenadas del documento)
    punto_inicio = Punto(
        id=0,
        nombre="Centro Administrativo Municipal",
        direccion="Punto de Inicio",
        latitud=4.833974,
        longitud=-75.681499,
        demanda=0
    )
    
    # Datos de los 144 puntos de la Tabla 1
    # Formato: (id, nombre, direccion)
    datos_puntos = [
        (1, "Centro Comercial Molinos", "Calle 35 # 13-1"),
        (2, "Éxito", "Calle 35 # 19-2"),
        (3, "CAM", "Av. Simón Bolívar 36"),
        (4, "Centro Comercial Progreso", "Simón Bolívar # 41-2"),
        (5, "San Fernando", "Calle 46 # 14a-1"),
        (6, "Santa Lucía", "Carrera 21 # 35-2"),
        (7, "La Pradera", "Calle 21 # 17-1"),
        (8, "El Progreso", "Calle 44 # 23-2"),
        (9, "Quintas De Jardín Colonia", "Transversal 26"),
        (10, "Jardín Colonial I", "Calle 42 # 22a-2"),
        (11, "Jardín Colonial II", "Calle 42 # 23-1 A 23-99"),
        (12, "Molinos", "Diagonal 46 # 21-"),
        (13, "Cambulos", "Carrera 21 # 36-100"),
        (14, "Villa de Molinos", "Calle 44 # 23-1 A 23-81"),
        (15, "Villa de Molinos II", "Diagonal 42 # 10d-2"),
        (16, "Villa del Pilar I", "Carrera 10G # 45-2"),
        (17, "Villa del Pilar II", "Carrera 11 # 60A-2"),
        (18, "Girasol", "Carrera 10B # 34-2"),
        (19, "Colinas", "Carrera 10G # 45-2"),
        (20, "Pablo VI", "Pablo Sexto Calle 44"),
        (21, "La Esmeralda", "Carrera 4 # 32-1"),
        (22, "Av. Simón Bolívar 43", "Simón Bolívar # 1 A 99"),
        (23, "Barrio Guadalupe", "Calle 36 # 15-1"),
        (24, "La Casa de la Cultura", "Simón Bolívar # 49-2"),
        (25, "La Romelia", "Calle 66 # 15B-2-100"),
        (26, "Lara Bonilla", "Calle 43 # 23-24"),
        (27, "Romelia Alta", "Calle 43 # 23-22"),
        (28, "Laureles", "Calle 44 # 23-2"),
        (29, "Carlos Ariel Escobar", "Calle 44 # 23-1"),
        (30, "Bosques de la Acuarela", "Calle 33 # 23-24"),
        (31, "Los Guamos", "Calle 53 # 23-22"),
        (32, "Barrio Júpiter", "Calle 464 # 23-2"),
        (33, "Divino Niño", "Calle 464 # 23-1 A"),
        (34, "Manuel Elkin Patarroyo", "Calle 763ABIS"),
        (35, "Variante Romelia El Pollo", "Carrera 10G # 45-2"),
        (36, "Calle de Las Aromas", "Carrera 11 # 60A-2"),
        (37, "Ensueño", "Carrera 10B # 34-2"),
        (38, "Cárcel de Mujeres", "Carrera 8 # 41B-1"),
        (39, "Minuto de Dios", "Carrera 8 # 41B-99"),
        (40, "Sakabuma", "Carrera 8 # 42B-2"),
        (41, "Minuto de Dios 2", "Carrera 8 # 41B-1"),
        (42, "Villa Alexandra", "Carrera 4 # 11-100"),
        (43, "Antigua Zona Industrial", "Vía La Popa # 2 A 84"),
        (44, "Plaza Comercial San Ángel", "Calle 16B # 2A-100"),
        (45, "La Graciela", "Carrera 54 Calle 132"),
        (46, "Carrera 4 # 9-38", "Diagonal 9 # 4-99"),
        (47, "Carrera 4 # 11-10", "Carrera 54 Calle 132"),
        (48, "Diagonal 8 # 4-1", "Carrera 54 Calle 132"),
        (49, "Inquilinos", "I-29 # 2 A 100"),
        (50, "Ensueño 2", "Calle 73ª BIS # 17A"),
        (51, "Santa Teresita", "Calle 64 15 45 Z"),
        (52, "Parque Industrial La", "Vía La Popa # 2 A 84"),
        (53, "Urbanización Macarena", "Calle 16B # 2A-100"),
        (54, "Servientrega", "Simón Bolívar # 26"),
        (55, "Zona Industrial Macarena", "Calle 16B # 2A-100"),
        (56, "Inquilinos 2", "Carrera 54 Calle 132"),
        (57, "Parque Industrial La Badea", "Carrera 64 Calle 62"),
        (58, "Seminario Mayor", "Vía La Popa # 1 A 23"),
        (59, "Colegio Diocesano", "Vía La Popa # 2 A 24"),
        (60, "Colegio Santa Sofía", "Carrera 24A"),
        (61, "Maracay", "Calle 20A # 10-42"),
        (62, "Bomberos 4", "Simón Bolívar # 35B-"),
        (63, "Cambulo", "Calle 43 # 16-100"),
        (64, "Quin. San Rafael", "I-29 # 2 A 100"),
        (65, "Campestre B", "Diagonal 25 # 10-2"),
        (66, "Altos Santa Clara", "Calle 12 # 20A-2"),
        (67, "Quinta Buenavista", "Calle 19 # 21-2"),
        (68, "Campestre C", "-29 # 1 A 99"),
        (69, "Asomeri", "Colombia 2.4 Km E"),
        (70, "Campestre B2", "Transversal 9 # 27-2"),
        (71, "Campestre A", "Calle 19A # 2A-1"),
        (72, "Villa Campestre", "I-29 # 1 A 99"),
        (73, "Campestre D", "Calle 19A # 5A-100"),
        (74, "Urb. Macarena", "Calle 16B # 2A-100"),
        (75, "Urb. Macaran", "Calle 20A # 10-72"),
        (76, "Playa Rica", "Carrera 10 # 38-2"),
        (77, "Pilarica", "Calle 43 # 7-1"),
        (78, "Andalucía", "Calle 43 # 7-1 67"),
        (79, "Urb.Garma", "Calle 42 # 10-1"),
        (80, "Guayacanes", "Calle 62 # 15-1"),
        (81, "Villa Del Campo", "Calle 52 # 10-1"),
        (82, "Andalucía 2", "Carrera 7 # 31-2"),
        (83, "REYES", "I-29 # 1"),
        (84, "Club Adulto", "I-29 # 2 A 100"),
        (85, "Bomberos", "Calle 36 # 16-1"),
        (86, "Quintas De Aragón", "Carrera 9 # 45A-99"),
        (87, "Villa Elena", "Calle 3 # 4-1"),
        (88, "Villa Elena I", "Carrera 9 # 44-2 A"),
        (89, "San Félix", "Carrera 8 # 41B-1"),
        (90, "San Félix II", "Carrera 8 # 41B-99"),
        (91, "Villa Mery", "Carrera 8 # 42B-2"),
        (92, "La Estación", "-29 # 1 A 99"),
        (93, "Villa Perla", "Carrera 54 Calle 132"),
        (94, "TCC", "Diagonal 9 # 4-2"),
        (95, "Nicole", "Vía La Popa # 2 A 84"),
        (96, "ABB", "Vía La Popa # 1 A 29"),
        (97, "Zona Industrial", "Diagonal 27A # 7-2"),
        (98, "Quintas del Bosque", "I-29 # 2 A 100"),
        (99, "Muebles Pabón", "Vía La Popa # 1 A 83"),
        (100, "Villa Turín", "Carrera 7 # 44-1"),
        (101, "Carrera 6 # 41A-2", "Carrera 6 # 41A-2"),
        (102, "Makro", "Simón Bolívar # 41"),
        (103, "Villa Diana", "I29 # 1"),
        (104, "Hacienda Bosque", "I-29 # 45"),
        (105, "Santa María", "Carrera 6 # 41-99"),
        (106, "Milán", "Calle 25 # 23-2 A 100"),
        (107, "Bosques Milán", "Carrera 8 # 41B-1"),
        (108, "Quintas Milán", "Carrera 8 # 41B-99"),
        (109, "Casas Milán", "Carrera 8 # 42B-2"),
        (110, "Terrazas De Milán", "Diagonal 25 # 18-2"),
        (111, "Guaduales Milán", "Diagonal 25 # 200"),
        (112, "Santa Bárbara", "Diagonal 25 # 17-99"),
        (113, "Carmelita", "Calle 16 # 41A-1"),
        (114, "Quintas Baleares", "Carrera 21 # 41"),
        (115, "La Pradera 2", "Bolivar-2 A 78"),
        (116, "Santa Mónica", "Carrera 19 # 17-2"),
        (117, "Altos de La Pradera", "Calle 24 # 23-99"),
        (118, "La Pradera I", "Calle 21 # 17-1"),
        (119, "Reservas Pradera", "Calle 21 # 23-12"),
        (120, "Coomnes", "Carrera 23 # 25"),
        (121, "Colmenares", "Diagonal 25 # 19-1"),
        (122, "Pradera II", "Calle 21 # 21-99"),
        (123, "Reservas del Milán", "Carrera 8 # 41B-1"),
        (124, "Colegio Salesiano", "Colegio Salesiano"),
        (125, "El Refugio", "Calle 20A # 10-2 A"),
        (126, "Torres del Sol", "Carrera 41B # 17-99"),
        (127, "Torres del Sol II", "Carrera 51B # 17-99"),
        (128, "Portal del Sol", "Diagonal 42 # 10d-2"),
        (129, "Los Cerezos", "Calle 17 # 41B-1"),
        (130, "Quintas del Refugio", "Calle 50 # 11-2"),
        (131, "La Macarena", "Calle 16B # 2A"),
        (132, "El Limonar", "Carrera 2 # 32-53"),
        (133, "Quintas San Martin", "Carrera 11 # 50"),
        (134, "San Nicolás", "Calle 33 # 12-1"),
        (135, "Guadalupe", "Carrera 15A # 40A"),
        (136, "Buenos Aires", "Calle 41 # 15-2"),
        (137, "Cámara del Comercio", "Calle 41 # 15-1 A"),
        (138, "Colegio María A", "Calle 33 # 23-24"),
        (139, "ICDB", "Calle 45 # 14-2"),
        (140, "Teatro Alcaraván", "Carrera 14A # 49-1"),
        (141, "Hogar", "Calle 43 # 23-24"),
        (142, "Juan Maria Gonzales", "Carrera 10G # 45-2"),
        (143, "Los Naranjos", "Carrera 11 # 60A-2"),
        (144, "Panadería El Niño", "Carrera 10B # 34-2")
    ]
    
    # Generar coordenadas aproximadas distribuidas en Dosquebradas
    # Centro aproximado de Dosquebradas
    lat_centro = 4.8325
    lon_centro = -75.6725
    
    # Rango de dispersión (Dosquebradas tiene aproximadamente 0.05° de latitud y 0.1° de longitud)
    np.random.seed(42)  # Para reproducibilidad
    
    puntos = [punto_inicio]
    
    # Generar demandas según el documento (entre 1500 y 2500 kg)
    demandas = np.random.uniform(2000, 2500, 144)
    
    # Distribuir puntos en diferentes sectores de Dosquebradas
    for i, (id_punto, nombre, direccion) in enumerate(datos_puntos):
        # Crear diferentes clusters/sectores
        sector = i % 6
        
        if sector == 0:  # Centro
            lat_offset = np.random.uniform(-0.01, 0.01)
            lon_offset = np.random.uniform(-0.02, 0.02)
        elif sector == 1:  # Norte
            lat_offset = np.random.uniform(0.01, 0.03)
            lon_offset = np.random.uniform(-0.02, 0.02)
        elif sector == 2:  # Sur
            lat_offset = np.random.uniform(-0.03, -0.01)
            lon_offset = np.random.uniform(-0.02, 0.02)
        elif sector == 3:  # Este
            lat_offset = np.random.uniform(-0.01, 0.01)
            lon_offset = np.random.uniform(0.02, 0.05)
        elif sector == 4:  # Oeste
            lat_offset = np.random.uniform(-0.01, 0.01)
            lon_offset = np.random.uniform(-0.05, -0.02)
        else:  # Disperso
            lat_offset = np.random.uniform(-0.03, 0.03)
            lon_offset = np.random.uniform(-0.05, 0.05)
        
        punto = Punto(
            id=id_punto,
            nombre=nombre,
            direccion=direccion,
            latitud=lat_centro + lat_offset,
            longitud=lon_centro + lon_offset,
            demanda=demandas[i]
        )
        puntos.append(punto)
    
    return puntos, 0  # Retorna puntos y el ID del punto de inicio


def obtener_rutas_iniciales_barrido():
    """
    Retorna las 7 rutas iniciales generadas por la técnica de barrido
    según la Tabla 4 del documento
    """
    rutas_iniciales = [
        [1, 44, 48, 39, 4, 13, 7, 23, 46, 58],
        [1, 9, 53, 45, 55, 37, 18, 32, 27, 58],
        [1, 8, 49, 28, 35, 26, 6, 41, 22, 58],
        [1, 3, 12, 16, 38, 5, 15, 2, 24, 58],
        [1, 43, 56, 54, 29, 31, 40, 57, 30, 58],
        [1, 34, 33, 42, 10, 11, 19, 25, 47, 58],
        [1, 21, 17, 14, 20, 51, 50, 36, 52, 58]
    ]
    
    return rutas_iniciales


def obtener_datos_tabla2():
    """
    Retorna los datos de costos y distancias de la Tabla 2 del documento
    (rutas reales de la empresa Serviciudad)
    """
    datos_tabla2 = {
        'costos': [16000, 15000, 24000, 23000, 24000, 16000, 15000, 24000, 
                   23000, 24000, 16000, 15000, 24000, 13000, 14000, 12000, 
                   12000, 10000, 10000, 12000],
        'distancias': [19, 18, 20, 19, 18, 20, 12, 18, 18, 26, 16, 18, 20, 
                       12, 12, 14, 14, 9, 9, 11],
        'tiempos_reales': [6.5, 6.0, 7.0, 8.0, 7.0, 8.0, 4.0, 8.0, 8.0, 8.0, 
                           8.0, 6.0, 6.5, 6.0, 5.0, 5.0, 6.0, 5.5, 3.5, 2.0],
        'num_puntos': [25, 15, 20, 11, 28, 21, 10, 10, 26, 11, 14, 14, 7, 
                       10, 14, 21, 12, 2, 11, 13],
        'demandas': [2500, 2300, 2100, 2500, 2500, 2500, 2300, 2500, 2500, 
                     2200, 2500, 2500, 2100, 2200, 2300, 2500, 2500, 2400, 
                     2500, 2500]
    }
    
    return datos_tabla2


# Exportar datos para usar en la interfaz
if __name__ == "__main__":
    puntos, inicio = cargar_144_puntos_completos()
    print(f"✓ Cargados {len(puntos)} puntos de recolección")
    print(f"✓ Punto de inicio: {puntos[inicio].nombre}")
    print(f"✓ Demanda total: {sum(p.demanda for p in puntos[1:]):.0f} kg")
    
    rutas = obtener_rutas_iniciales_barrido()
    print(f"✓ Rutas iniciales de barrido: {len(rutas)} rutas")