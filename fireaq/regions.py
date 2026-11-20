"""Study regions for v0.3 (frozen in docs/PREREGISTRATION_v0.3.md)."""
REGIONS = {
    'norcal':    {'bbox': (-123.5, 38.5, -119.5, 42.5), 'role': 'development',
                  'name': 'Northern California'},
    'nrockies':  {'bbox': (-117.0, 44.0, -113.0, 48.0), 'role': 'development',
                  'name': 'Central Idaho / W Montana'},
    'southwest': {'bbox': (-112.0, 32.0, -108.0, 36.0), 'role': 'development',
                  'name': 'SE Arizona / SW New Mexico'},
    'southeast': {'bbox': (-85.0, 30.0, -81.0, 34.0), 'role': 'development',
                  'name': 'S Georgia / N Florida'},
    'pnw':       {'bbox': (-123.0, 43.0, -119.0, 47.0), 'role': 'test_space',
                  'name': 'Oregon / Washington Cascades'},
    'ozarks':    {'bbox': (-96.0, 34.0, -92.0, 38.0), 'role': 'test_space',
                  'name': 'E Oklahoma / W Arkansas / SW Missouri'},
}
STEP = 0.5
DEV_REGIONS = [k for k, v in REGIONS.items() if v['role'] == 'development']
TEST_SPACE_REGIONS = [k for k, v in REGIONS.items() if v['role'] == 'test_space']
PREVIOUSLY_TESTED = {'norcal': ('2024-01-01', '2025-12-31')}   # v0.2 test period: never "untouched"
