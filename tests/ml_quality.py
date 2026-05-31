# import json
# import pytest
# from lambda_registrar_experiments import lambda_handler

# TEST_CASES = [
#     {"name": "zurich", "event": {"carrera": "zurich", "splits": ["km5"]}},
#     {"name": "valencia", "event": {"carrera": "valencia", "splits": ["km21"]}},
#     {"name": "madrid", "event": {"carrera": "madrid", "splits": ["km5"]}}
# ]

# @pytest.mark.parametrize("test_case", TEST_CASES)
# def test_ml_quality_end_to_end(test_case):
#     print(f"\n🧪 {test_case['name']}")
    
#     result = lambda_handler(test_case['event'])
#     assert result['statusCode'] == 200
    
#     body = json.loads(result['body'])
    
#     # MÍNIMO funcional (tu lambda actual)
#     assert 'num_modelos' in body
#     assert body['num_modelos'] >= 1, f"0 modelos: {test_case['name']}"
    
#     mae = body.get('metrics', {}).get('mae', 999)
#     assert mae < 60, f"MAE={mae}s fail"
    
#     print(f"✅ {test_case['name']}: {body['num_modelos']} modelos")