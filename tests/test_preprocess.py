import pytest
import pandas as pd
import numpy as np
from src.preprocessor import CustomFeatureEngineer

@pytest.fixture
def sample_data():
    data = {
        'tpep_pickup_datetime': ['2023-01-01 10:00:00', '2023-01-02 15:30:00'],
        'tpep_dropoff_datetime': ['2023-01-01 10:30:00', '2023-01-02 16:00:00'],
        'trip_distance': [5.0, 10.0],
        'duration': [1800, 1800],
        'PULocationID': [1, 2],
        'DOLocationID': [3, 4],
        'Borough_pickup': ['Manhattan', 'Brooklyn'],
        'Borough_dropoff': ['Queens', 'Manhattan'],
        'service_zone_pickup': ['Yellow', 'Green'],
        'service_zone_dropoff': ['Green', 'Yellow'],
        'fare_amount': [15.0, 25.0]
    }
    return pd.DataFrame(data)

def test_time_features(sample_data):
    feature_engineer = CustomFeatureEngineer()
    
    sample_data['tpep_pickup_datetime'] = pd.to_datetime(sample_data['tpep_pickup_datetime'])
    result = feature_engineer._time_features(sample_data.copy())

    assert 'pu_hour_sin' in result.columns
    assert 'pu_hour_cos' in result.columns
    assert 'pu_minute_sin' in result.columns
    assert 'pu_minute_cos' in result.columns
    assert 'pu_weekday_sin' in result.columns
    assert 'pu_weekday_cos' in result.columns
    assert 'is_weekend' in result.columns

    # Check specific values
    assert np.isclose(result['pu_hour_sin'][0], np.sin(2 * np.pi * 10 / 24))
    assert result['is_weekend'][0] == 1  # Not a weekend

def test_additional_features(sample_data):
    feature_engineer = CustomFeatureEngineer()
    result = feature_engineer._additional_features(sample_data.copy())

    assert 'speed' in result.columns
    assert 'id_route' in result.columns
    assert 'borough_route' in result.columns
    assert 'zone_route' in result.columns
    assert 'same_borough' in result.columns
    assert 'same_service_zone' in result.columns

    # Check specific values
    assert result['speed'][0] == 10.0  # 5 miles in 0.5 hours
    assert result['same_borough'][0] == 0  # Manhattan != Queens
    assert result['same_service_zone'][0] == 0  # Yellow != Green
