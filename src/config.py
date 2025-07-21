import os

# Compute project root relative to this config file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "output.log")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# data urls    
DATA_URLS = ['https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet',
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet',
        'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet',
        'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv']

# NA dictionary
NA_DICT = {'passenger_count': 1,
           'RatecodeID': 99,
           'store_and_fwd_flag': 'Z',
           'Borough_pickup': 'Outside of NYC',
           'service_zone_pickup': 'Unknown',
           'Borough_dropoff': 'Outside of NYC',
           'service_zone_dropoff': 'Unknown'}

# Train and validation split
TRAIN_CUT_OFF_DATE = '2023-03-17'
VAL_CUT_OFF_DATE = '2023-03-24'
