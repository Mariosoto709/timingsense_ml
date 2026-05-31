# """
# tests/unit/test_glue_utils.py
# Pruebas unitarias para funciones auxiliares de Glue
# """

# import sys
# import os
# import pytest
# from unittest.mock import patch, MagicMock, Mock

# # Importar desde glue_utils
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../glue')))

# from glue_utils import (
#     extract_split_distance,
#     splits_son_equivalentes,
#     get_split_type,
#     find_closest_split,
#     analyze_split_requirements,
#     cargar_catalogo_distancias,
#     listar_carreras_disponibles,
#     calcular_cobertura_carrera,
#     buscar_mejor_combinacion_fallback
# )


# # =============================================================
# # TESTS PARA cargar_catalogo_distancias
# # =============================================================

# class TestCargarCatalogoDistancias:
#     """Pruebas para cargar_catalogo_distancias (con mock de S3)"""
    
#     def test_carga_exitosa(self, monkeypatch):
#         """✅ Caso feliz: el catálogo existe y es válido"""
#         import json
        
#         # Mock de la respuesta de S3
#         mock_s3 = Mock()
#         mock_s3.get_object.return_value = {
#             'Body': Mock(read=lambda: json.dumps({
#                 'distancias': {'km_5': 5000, 'km_10': 10000}
#             }).encode())
#         }
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         resultado = cargar_catalogo_distancias('test_race')
        
#         assert resultado == {'distancias': {'km_5': 5000, 'km_10': 10000}}
#         mock_s3.get_object.assert_called_once_with(
#             Bucket='timingsense-config',
#             Key='splits_catalog/distancias/test_race.json'
#         )
    
#     def test_catalogo_no_existe(self, monkeypatch):
#         """❌ Carrera sin catálogo → retorna None"""
#         mock_s3 = Mock()
#         mock_s3.get_object.side_effect = Exception('NoSuchKey')
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         resultado = cargar_catalogo_distancias('race_inexistente')
        
#         assert resultado is None
    
#     def test_bucket_personalizado(self, monkeypatch):
#         """🎯 Usar bucket diferente al default"""
#         import json
#         mock_s3 = Mock()
#         mock_s3.get_object.return_value = {
#             'Body': Mock(read=lambda: json.dumps({}).encode())
#         }
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         cargar_catalogo_distancias('test', bucket='mi-bucket', prefix='custom/')
        
#         mock_s3.get_object.assert_called_with(
#             Bucket='mi-bucket',
#             Key='custom/test.json'
#         )


# # =============================================================
# # TESTS PARA listar_carreras_disponibles
# # =============================================================

# class TestListarCarrerasDisponibles:
#     """Pruebas para listar_carreras_disponibles"""
    
#     def test_lista_con_varias_carreras(self, monkeypatch):
#         """📋 Múltiples carreras disponibles"""
#         mock_s3 = Mock()
#         mock_s3.list_objects_v2.return_value = {
#             'Contents': [
#                 {'Key': 'splits_catalog/distancias/barcelona_2024.json'},
#                 {'Key': 'splits_catalog/distancias/madrid_2024.json'},
#                 {'Key': 'splits_catalog/distancias/valencia_2023.json'},
#                 {'Key': 'splits_catalog/distancias/otros/ignorado.json'},  
#             ]
#         }
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         resultado = listar_carreras_disponibles()
        
#         assert set(resultado) == {'barcelona_2024', 'madrid_2024', 'valencia_2023'}
#         assert len(resultado) == 3
    
#     def test_lista_vacia(self, monkeypatch):
#         """📭 No hay carreras disponibles"""
#         mock_s3 = Mock()
#         mock_s3.list_objects_v2.return_value = {'Contents': []}
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         resultado = listar_carreras_disponibles()
        
#         assert resultado == []
    
#     def test_error_s3(self, monkeypatch):
#         """⚠️ Error de conexión a S3"""
#         mock_s3 = Mock()
#         mock_s3.list_objects_v2.side_effect = Exception('ConnectionError')
        
#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)
        
#         resultado = listar_carreras_disponibles()
        
#         assert resultado == []  # Retorna lista vacía en caso de error
    
#     def test_prefix_personalizado(self, monkeypatch):
#         """🎯 Usar prefijo diferente"""
#         mock_s3 = Mock()
#         mock_s3.list_objects_v2.return_value = {'Contents': []}

#         monkeypatch.setattr('boto3.client', lambda *args, **kwargs: mock_s3)

#         listar_carreras_disponibles(prefix='custom/path/')

#         # ✅ CORREGIDO: Incluir Delimiter que ahora usa la función
#         mock_s3.list_objects_v2.assert_called_with(
#             Bucket='timingsense-config',
#             Prefix='custom/path/',
#             Delimiter='/'  # ← Añadir esto
#         )


# # =============================================================
# # TESTS PARA calcular_cobertura_carrera
# # =============================================================

# class TestCalcularCoberturaCarrera:
#     """Pruebas para calcular_cobertura_carrera"""
    
#     @pytest.fixture
#     def catalogo_maraton_completo(self):
#         """Fixture: Maratón con splits estándar"""
#         return {
#             'splits': [
#                 {'event': 'Marató', 'distance': 5000},
#                 {'event': 'Marató', 'distance': 10000},
#                 {'event': 'Marató', 'distance': 15000},
#                 {'event': 'Marató', 'distance': 21097.5},
#                 {'event': 'Marató', 'distance': 30000},
#                 {'event': 'Marató', 'distance': 42195},
#             ]
#         }
    
#     @pytest.fixture
#     def catalogo_multi_evento(self):
#         """Fixture: Múltiples eventos en una carrera"""
#         return {
#             'splits': [
#                 {'event': '10K', 'distance': 5000},
#                 {'event': '10K', 'distance': 10000},
#                 {'event': 'Mitja', 'distance': 5000},
#                 {'event': 'Mitja', 'distance': 10000},
#                 {'event': 'Mitja', 'distance': 15000},
#                 {'event': 'Mitja', 'distance': 21097.5},
#             ]
#         }
        
#     def test_cobertura_100_por_ciento(self, catalogo_maraton_completo):
#         """🎯 Todos los puntos cubiertos exactamente"""
#         puntos = [5000, 10000, 15000, 21097.5, 42195]
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos)
        
#         assert resultado['cobertura'] == 1.0
#         assert len(resultado['puntos_cubiertos']) == 5
#         assert len(resultado['puntos_faltantes']) == 0
#         assert resultado['evento_elegido'] == 'Marató'
    
#     def test_cobertura_parcial(self, catalogo_maraton_completo):
#         """🎯 Solo algunos puntos están disponibles"""
#         puntos = [5000, 12345, 25000, 42195]  # 12345 y 25000 no existen
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos)
        
#         assert resultado['cobertura'] == 0.5  # 2 de 4 puntos
#         assert set(resultado['puntos_cubiertos']) == {5000, 42195}
#         assert set(resultado['puntos_faltantes']) == {12345, 25000}
    
#     def test_sin_cobertura(self, catalogo_maraton_completo):
#         """🎯 Ningún punto está disponible"""
#         puntos = [12345, 23456, 34567]
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos)
        
#         assert resultado['cobertura'] == 0.0
#         assert len(resultado['puntos_cubiertos']) == 0
#         assert len(resultado['puntos_faltantes']) == 3
    
#     # ----- Tests de tolerancia -----
    
#     def test_tolerancia_10_metros(self, catalogo_maraton_completo):
#         """📏 Permite diferencias de hasta 10 metros"""
#         puntos = [5005, 10010, 15015]  
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos, tolerancia=10)
        
#         # 5005 está a 5m de 5000 → cubierto
#         # 10010 está a 10m de 10000 → cubierto (límite)
#         # 15015 está a 15m de 15000 → NO cubierto
#         assert len(resultado['puntos_cubiertos']) == 2
#         assert 15015 in resultado['puntos_faltantes']
    
#     def test_tolerancia_20_metros(self, catalogo_maraton_completo):
#         """📏 Con tolerancia mayor, cubre más puntos"""
#         puntos = [5005, 10010, 15015]
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos, tolerancia=20)
        
#         assert len(resultado['puntos_cubiertos']) == 3  # Todos cubiertos
    
#     # ----- Tests de filtrado por evento -----
    
#     def test_filtro_evento_especifico(self, catalogo_multi_evento):
#         """🎯 Filtrar por '10K' solo usa esos splits"""
#         puntos = [5000, 10000, 15000]  # 15000 solo está en Mitja
        
#         resultado = calcular_cobertura_carrera(
#             catalogo_multi_evento, puntos, event_id_filter='10K'
#         )
        
#         assert resultado['evento_elegido'] == '10K'
#         assert resultado['cobertura'] == 2/3  # 5000 y 10000 sí, 15000 no
#         assert 15000 in resultado['puntos_faltantes']
    
#     def test_filtro_evento_case_insensitive(self, catalogo_multi_evento):
#         """🔤 El filtro debe ser case-insensitive"""
#         puntos = [5000, 10000]
        
#         resultado = calcular_cobertura_carrera(
#             catalogo_multi_evento, puntos, event_id_filter='mitja'
#         )
        
#         assert resultado['evento_elegido'] == 'Mitja'
#         assert resultado['cobertura'] == 1.0
    
#     def test_filtro_evento_inexistente(self, catalogo_multi_evento):
#         """⚠️ Evento no encontrado → usa todos como fallback"""
#         puntos = [5000, 10000]
        
#         resultado = calcular_cobertura_carrera(
#             catalogo_multi_evento, puntos, event_id_filter='Inexistente'
#         )
        
#         # Como fallback, usa todos los eventos y encuentra cobertura
#         assert resultado['cobertura'] == 1.0
#         assert resultado['evento_elegido'] in ['10K', 'Mitja']
    
#     # ----- Tests de selección del mejor evento -----
    
#     def test_selecciona_evento_con_mejor_cobertura(self, catalogo_multi_evento):
#         """🏆 Elige el evento que da mayor cobertura"""
#         # 10K tiene: 5000, 10000
#         # Mitja tiene: 5000, 10000, 15000, 21097.5
#         puntos = [5000, 10000, 15000]  # Mitja cubre 3, 10K cubre 2
        
#         resultado = calcular_cobertura_carrera(catalogo_multi_evento, puntos)
        
#         assert resultado['evento_elegido'] == 'Mitja'
#         assert resultado['cobertura'] == 1.0
    
#     def test_empate_cobertura(self, catalogo_multi_evento):
#         """🤝 Empate: elige el primero que encuentra (orden del catálogo)"""
#         puntos = [5000, 10000]  # Ambos eventos cubren exactamente igual
        
#         resultado = calcular_cobertura_carrera(catalogo_multi_evento, puntos)
        
#         # El primero en aparecer (10K) debería ser elegido
#         assert resultado['evento_elegido'] == '10K'
    
#     # ----- Tests con formato antiguo (sin 'splits') -----
    
#     def test_catalogo_formato_legacy(self):
#         """🔄 Compatibilidad con catálogos que usan 'distancias' en lugar de 'splits'"""
#         catalogo_legacy = {
#             'distancias': {
#                 'km_5': 5000,
#                 'km_10': 10000,
#                 'km_15': 15000,
#             }
#         }
#         puntos = [5000, 10000]
        
#         resultado = calcular_cobertura_carrera(catalogo_legacy, puntos)
        
#         assert resultado['cobertura'] == 1.0
#         assert resultado['evento_elegido'] is None  # 'todos' se convierte en None
    
#     # ----- Tests de edge cases -----
    
#     def test_catalogo_vacio(self):
#         """📭 Catálogo sin splits"""
#         catalogo_vacio = {'splits': []}
#         puntos = [5000, 10000]
        
#         resultado = calcular_cobertura_carrera(catalogo_vacio, puntos)
        
#         assert resultado['cobertura'] == 0.0
#         assert resultado['puntos_cubiertos'] == []
#         assert resultado['evento_elegido'] is None
    
#     def test_lista_puntos_vacia(self, catalogo_maraton_completo):
#         """📭 Sin puntos que evaluar"""
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, [])
        
#         assert resultado['cobertura'] == 0.0  # 0/0? debería ser 0
#         assert resultado['puntos_cubiertos'] == []
#         assert resultado['puntos_faltantes'] == []
    
#     def test_puntos_con_duplicados(self, catalogo_maraton_completo):
#         """🔄 Puntos duplicados no deben afectar"""
#         puntos = [5000, 5000, 10000, 10000]
        
#         resultado = calcular_cobertura_carrera(catalogo_maraton_completo, puntos)
        
#         # Los duplicados se procesan pero la cobertura es sobre puntos únicos
#         assert resultado['cobertura'] == 1.0
#         assert len(resultado['puntos_cubiertos']) == 4  # Todos los puntos (incluye duplicados)


# # =============================================================
# # TESTS PARA buscar_mejor_combinacion_fallback
# # =============================================================

# class TestBuscarMejorCombinacionFallback:
#     """Pruebas para buscar_mejor_combinacion_fallback"""
    
#     @pytest.fixture
#     def candidatas_ejemplo(self):
#         """Fixture: 3 carreras con diferentes coberturas"""
#         return [
#             {
#                 'race_id': 'carrera_A',
#                 'cobertura': 0.8,
#                 'puntos_cubiertos': [5000, 10000, 15000]
#             },
#             {
#                 'race_id': 'carrera_B',
#                 'cobertura': 0.7,
#                 'puntos_cubiertos': [15000, 21097, 30000]
#             },
#             {
#                 'race_id': 'carrera_C',
#                 'cobertura': 0.6,
#                 'puntos_cubiertos': [30000, 42195]
#             }
#         ]
    
#     @pytest.fixture
#     def puntos_completos(self):
#         """Fixture: 6 puntos que requieren todas las carreras"""
#         return [5000, 10000, 15000, 21097, 30000, 42195]
    
#     # ----- Tests de combinaciones básicas -----
    
#     def test_combinacion_perfecta(self, candidatas_ejemplo, puntos_completos):
#         """🎯 Combinación de 3 carreras cubre el 100%"""
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos_completos, max_carreras=3
#         )
        
#         assert resultado['cobertura_total'] == 1.0
#         assert set(resultado['seleccionadas']) == {'carrera_A', 'carrera_B', 'carrera_C'}
#         assert len(resultado['puntos_cubiertos']) == 6
#         assert len(resultado['puntos_faltantes']) == 0
    
#     def test_max_carreras_limitado(self, candidatas_ejemplo, puntos_completos):
#         """🔢 Con max_carreras=2, no puede cubrir todo"""
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos_completos, max_carreras=2
#         )
        
#         # Carrera_A + Carrera_B cubren: 5000,10000,15000,21097,30000 (5 puntos)
#         # Falta 42195
#         assert resultado['cobertura_total'] == 5/6
#         assert len(resultado['seleccionadas']) <= 2
#         assert 42195 in resultado['puntos_faltantes']
    
#     def test_elige_mejor_combinacion(self, candidatas_ejemplo):
#         """🏆 Elige la combinación que maximiza cobertura"""
#         puntos = [5000, 10000, 30000, 42195]
        
#         # Opciones:
#         # - Solo A: cubre 5000,10000 (50%)
#         # - Solo C: cubre 30000,42195 (50%)
#         # - A + C: cubre 100% (mejor)
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos, max_carreras=2
#         )
        
#         assert resultado['cobertura_total'] == 1.0
#         assert set(resultado['seleccionadas']) == {'carrera_A', 'carrera_C'}
    
#     # ----- Tests de umbral mínimo -----
    
#     def test_umbral_minimo_filtra_candidatas(self, candidatas_ejemplo):
#         """🎯 Solo considera carreras con cobertura >= umbral"""
#         puntos = [5000, 10000, 15000]
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos, umbral_minimo=0.75
#         )
        
#         # Carrera_A (0.8) pasa, Carrera_B (0.7) y C (0.6) no
#         assert resultado['seleccionadas'] == ['carrera_A']
#         assert resultado['cobertura_total'] == 1.0
    
#     def test_umbral_muy_alto_sin_resultados(self, candidatas_ejemplo):
#         """⚠️ Umbral > todas las candidatas → resultado vacío"""
#         puntos = [5000, 10000]
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos, umbral_minimo=0.95
#         )
        
#         assert resultado['seleccionadas'] == []
#         assert resultado['cobertura_total'] == 0.0
#         assert len(resultado['puntos_faltantes']) == 2
    
#     # ----- Tests de casos borde -----
    
#     def test_lista_candidatas_vacia(self):
#         """📭 Sin candidatas → sin cobertura"""
#         resultado = buscar_mejor_combinacion_fallback([], [5000, 10000])
        
#         assert resultado['seleccionadas'] == []
#         assert resultado['cobertura_total'] == 0.0
#         assert len(resultado['puntos_faltantes']) == 2
    
#     def test_una_sola_candidata_suficiente(self, candidatas_ejemplo):
#         """✅ Una carrera ya cubre todo"""
#         puntos = [5000, 10000, 15000]
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos, max_carreras=3
#         )
        
#         assert resultado['seleccionadas'] == ['carrera_A']
#         assert resultado['cobertura_total'] == 1.0
    
#     def test_sin_cobertura_total(self, candidatas_ejemplo):
#         """❌ Imposible cubrir todos los puntos"""
#         puntos = [9999, 8888, 7777]  # Puntos que ninguna carrera tiene
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos, max_carreras=3
#         )
        
#         assert resultado['cobertura_total'] == 0.0
#         assert resultado['seleccionadas'] == []
#         assert len(resultado['puntos_faltantes']) == 3
    
#     def test_puntos_duplicados(self, candidatas_ejemplo):
#         """🔄 Puntos duplicados no afectan el cálculo"""
#         puntos = [5000, 5000, 10000, 10000, 15000]
        
#         resultado = buscar_mejor_combinacion_fallback(
#             candidatas_ejemplo, puntos
#         )
        
#         # La cobertura se calcula sobre puntos únicos
#         assert resultado['cobertura_total'] == 1.0
#         assert len(resultado['puntos_cubiertos']) == 5  # Incluye duplicados
    
#     # ----- Test de rendimiento (opcional) -----
    
#     def test_combinaciones_muchas_candidatas(self):
#         """⚡ Prueba que el algoritmo escala (no combinatoria explosiva)"""
#         # Crear 10 candidatas con coberturas variadas
#         candidatas = []
#         for i in range(10):
#             candidatas.append({
#                 'race_id': f'carrera_{i}',
#                 'cobertura': 0.5 + i * 0.05,
#                 'puntos_cubiertos': [1000 * i, 2000 * i]
#             })
        
#         puntos = [1000, 2000, 3000, 4000, 5000]
        
#         # No debe tardar mucho (max_carreras=3 → C(10,3)=120 combinaciones)
#         import time
#         start = time.time()
#         resultado = buscar_mejor_combinacion_fallback(candidatas, puntos, max_carreras=3)
#         elapsed = time.time() - start
        
#         assert elapsed < 0.1  # Menos de 100ms
#         assert 'cobertura_total' in resultado

# # =============================================================
# # TESTS PARA extract_split_distance (ya existen, los mantenemos)
# # =============================================================

# class TestExtractSplitDistance:
#     """Pruebas para extract_split_distance"""
    
#     @pytest.mark.parametrize("split,esperado", [
#         ("km_5", 5.0),
#         ("km_10", 10.0),
#         ("km_42", 42.0),
#         ("km_18_2", 18.2), 
#         ("km_21_0975", 21.0975),
#         ("km_18.2", 18.2),
#         ("half", 21.0975),
#         ("finish", 42.195),
#         ("start", 0.0),
#         (None, None),
#         ("", None),
#         ("km_", None)
#     ])
#     def test_extract_split_distance_completo(self, split, esperado):
#         """🎯 MEJORA: Parametriza TODOS los casos (DRY)"""
#         assert extract_split_distance(split) == esperado


# # =============================================================
# # TESTS PARA splits_son_equivalentes
# # =============================================================

# class TestSplitsSonEquivalentes:
#     """Pruebas para splits_son_equivalentes"""
    
#     def test_mismo_nombre(self):
#         assert splits_son_equivalentes("km_5", "km_5") == True
    
#     def test_punto_vs_guion(self):
#         assert splits_son_equivalentes("km_18_2", "km_18.2") == True
    
#     def test_half_equivalencia(self):
#         assert splits_son_equivalentes("half", "km_21.0975") == True
    
#     def test_finish_equivalencia(self):
#         assert splits_son_equivalentes("finish", "km_42.195") == True
    
#     def test_no_equivalentes(self):
#         assert splits_son_equivalentes("km_5", "km_10") == False


# # =============================================================
# # TESTS PARA get_split_type
# # =============================================================

# class TestGetSplitType:
#     """Pruebas para get_split_type"""
    
#     def test_get_split_type_distance_km(self):
#         assert get_split_type("km_5") == "distance"
#         assert get_split_type("km_10") == "distance"
    
#     def test_get_split_type_distance_half(self):
#         assert get_split_type("half") == "distance"
    
#     def test_get_split_type_distance_finish(self):
#         assert get_split_type("finish") == "distance"
    
#     def test_get_split_type_other(self):
#         assert get_split_type("nombre_split") == "other"
#         assert get_split_type(None) == "other"


# # =============================================================
# # TESTS PARA find_closest_split
# # =============================================================

# class TestFindClosestSplit:
#     """Pruebas para find_closest_split"""
    
#     def test_find_closest_split_exact(self):
#         historical_splits = ["km_5", "km_10", "km_15"]
#         closest, dist = find_closest_split(historical_splits, 10.0)
#         assert closest == "km_10"
#         assert dist == 10.0
    
#     def test_find_closest_split_approximate(self):
#         historical_splits = ["km_5", "km_10", "km_15"]
#         closest, dist = find_closest_split(historical_splits, 12.0)
#         assert closest == "km_10"
#         assert dist == 10.0
    
#     def test_find_closest_split_empty_list(self):
#         result = find_closest_split([], 10.0)
#         assert result == (None, None)


# # =============================================================
# # TESTS PARA analyze_split_requirements
# # =============================================================

# class TestAnalyzeSplitRequirements:
#     """Pruebas para analyze_split_requirements"""
    
#     @pytest.fixture
#     def carreras_historicas_sample(self):
#         return [
#             {"splits": ["km_5", "km_10", "half", "finish"]},
#             {"splits": ["km_5", "km_10", "km_15", "half", "finish"]}
#         ]
    
#     def test_analyze_split_requirements_direct(self, carreras_historicas_sample):
#         splits_objetivo = ["km_5", "km_10"]
#         resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
#         assert "km_5" in resultado["splits_directos"]
#         assert "km_10" in resultado["splits_directos"]
#         assert len(resultado["splits_interpolables"]) == 0
    
#     def test_analyze_split_requirements_interpolate(self, carreras_historicas_sample):
#         splits_objetivo = ["km_18_2"]
#         resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
#         assert len(resultado["splits_directos"]) == 0
#         assert len(resultado["splits_interpolables"]) == 1
#         assert resultado["mapping"]["km_18_2"][0] == "interpolate"
    
#     def test_analyze_split_requirements_impossible(self, carreras_historicas_sample):
#         splits_objetivo = ["split_inexistente"]
#         resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
#         assert "split_inexistente" in resultado["splits_imposibles"]
#         assert len(resultado["splits_finales"]) == 0

#     def test_analyze_split_requirements_realista(self, carreras_historicas_sample):
#         """Barcelona-2026 pide km_5+km_12+half (mix directo/interpolar)"""
#         splits_objetivo = ["km_5", "km_12", "half", "split_imposible"]
#         resultado = analyze_split_requirements(splits_objetivo, carreras_historicas_sample)
        
#         assert len(resultado["splits_directos"]) == 2        # km_5, half  
#         assert len(resultado["splits_interpolables"]) == 1    # km_12
#         assert len(resultado["splits_imposibles"]) == 1       # split_imposible
#         assert len(resultado["splits_finales"]) == 3          # km_5, km_12, half


# import pandas as pd
# import numpy as np
# from glue_utils import validar_calidad_datos


# class TestValidarCalidadDatos:
#     """Pruebas para validar_calidad_datos"""
    
#     def test_validacion_datos_correctos(self):
#         """Datos correctos deben pasar la validación"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': np.random.normal(1800, 100, 200),
#             'km_10': np.random.normal(3600, 150, 200),
#         })
#         splits = ['km_5', 'km_10']
        
#         resultado = validar_calidad_datos(df, splits)
        
#         assert resultado['valido'] == True
#         assert len(resultado['errores']) == 0
#         assert resultado['metricas']['n_registros'] == 200
    
#     def test_validacion_pocos_registros(self):
#         """Pocos registros debe fallar"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(50)],
#             'km_5': np.random.normal(1800, 100, 50),
#         })
#         splits = ['km_5']
        
#         resultado = validar_calidad_datos(df, splits, umbral_min_registros=100)
        
#         assert resultado['valido'] == False
#         assert "Registros insuficientes" in resultado['errores'][0]
#         assert resultado['metricas']['n_registros'] == 50
    
#     def test_validacion_nulos_excesivos_error(self):
#         """Nulos excesivos (>50%) debe fallar"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': [np.nan if i < 150 else 1800 for i in range(200)],  # 75% nulos
#             'km_10': np.random.normal(3600, 150, 200),
#         })
#         splits = ['km_5', 'km_10']
        
#         resultado = validar_calidad_datos(df, splits, umbral_nulos_error=0.5)
        
#         assert resultado['valido'] == False
#         assert "nulos" in resultado['errores'][0]
#         assert resultado['metricas']['nulos_km_5'] > 0.7
    
#     def test_validacion_nulos_warning(self):
#         """Nulos moderados (20-50%) genera warning pero no falla"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': [np.nan if i < 60 else 1800 for i in range(200)],  # 30% nulos
#             'km_10': np.random.normal(3600, 150, 200),
#         })
#         splits = ['km_5', 'km_10']
        
#         resultado = validar_calidad_datos(df, splits, 
#                                           umbral_nulos_warning=0.2,
#                                           umbral_nulos_error=0.5)
        
#         # Debe pasar (no error), pero tener warning
#         assert resultado['valido'] == True
#         assert len(resultado['warnings']) > 0
#         assert "nulos" in resultado['warnings'][0]
    
#     def test_validacion_split_inexistente(self):
#         """Split que no existe en los datos debe fallar"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': np.random.normal(1800, 100, 200),
#         })
#         splits = ['km_5', 'km_inexistente']
        
#         resultado = validar_calidad_datos(df, splits)
        
#         assert resultado['valido'] == False
#         assert "no encontrado" in resultado['errores'][0]
    
#     def test_validacion_tiempos_negativos_warning(self):
#         """Tiempos negativos generan warning"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': list(np.random.normal(1800, 100, 195)) + [-100, -200, -300, -400, -500],
#         })
#         splits = ['km_5']
        
#         resultado = validar_calidad_datos(df, splits)
        
#         assert len(resultado['warnings']) > 0
#         assert "negativos" in resultado['warnings'][0]
#         assert resultado['valido'] == True  # No falla, solo warning
    
#     def test_validacion_tiempos_muy_grandes_warning(self):
#         """Tiempos > 6 horas generan warning"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': list(np.random.normal(1800, 100, 195)) + [25000, 30000, 35000, 40000, 50000],
#         })
#         splits = ['km_5']
        
#         resultado = validar_calidad_datos(df, splits, tiempo_maximo_segundos=21600)
        
#         assert len(resultado['warnings']) > 0
#         assert "horas" in resultado['warnings'][0]
#         assert resultado['valido'] == True
    
#     def test_validacion_todo_mal(self):
#         """Múltiples problemas: pocos registros, nulos, split inexistente"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(30)],
#             'km_5': [np.nan] * 30,  # todos nulos
#         })
#         splits = ['km_5', 'km_inexistente']
        
#         resultado = validar_calidad_datos(df, splits, umbral_min_registros=100)
        
#         assert resultado['valido'] == False
#         assert len(resultado['errores']) >= 2  
    
#     def test_validacion_metricas_retornadas(self):
#         """Verifica que las métricas se retornan correctamente"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': [1800] * 200,
#             'km_10': [3600] * 200,
#         })
#         splits = ['km_5', 'km_10']
        
#         resultado = validar_calidad_datos(df, splits)
        
#         assert 'n_registros' in resultado['metricas']
#         assert 'n_splits' in resultado['metricas']
#         assert 'nulos_km_5' in resultado['metricas']
#         assert 'nulos_km_10' in resultado['metricas']

#     def test_validacion_outliers_maraton(self):
#         """5% corredores imposibles (>6h km_5) → WARNING"""
#         df = pd.DataFrame({
#             'athlete_id': [f'a{i}' for i in range(200)],
#             'km_5': [1800]*190 + [25000, 30000, 35000, 40000, 50000, 55000, 60000, 65000, 70000, 75000]
#         })
#         resultado = validar_calidad_datos(df, ['km_5'])
#         assert len(resultado['warnings']) > 0
#         assert "horas" in resultado['warnings'][0]
#         assert resultado['valido'] == True


# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])