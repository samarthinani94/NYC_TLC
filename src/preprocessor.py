import pandas as pd
import numpy as np
import os
import requests
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, NA_DICT, DATA_URLS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.logger import get_logger
from time import sleep

logger = get_logger(__name__)


class DataLoader:
    """Loads the taxi data and the zone lookup data, and joins them into a single DataFrame."""
    def __init__(self, raw_data_dir, data_urls, fetch_new_data=False):
        self.data_dir = raw_data_dir
        self.data_urls = data_urls
        self.fetch_new_data = fetch_new_data
            
    def _fetch_data_from_nyc_tlc_website(self):
        """Fetches the latest taxi data from the NYC TLC website."""
        logger.info("Fetching data from NYC TLC website.")
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        for url in self.data_urls:
            file_name = url.split("/")[-1]
            file_path = os.path.join(self.data_dir, file_name)

            
            # Check if the file already exists
            if os.path.exists(file_path):
                logger.info(f"{file_name} already exists. Skipping download.")
                continue
            # Download the file
            logger.info(f"Downloading {file_name}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Downloaded {file_name} to {file_path}")
            else:
                logger.warning(f"Failed to download {file_name}. Status code: {response.status_code}")

    def _load_taxi_data(self):
        logger.info("Starting to load taxi data.")
        """Load all parquet files from the specified directory into a single DataFrame."""
        dataframes = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".parquet"):
                file_path = os.path.join(self.data_dir, file_name)
                df = pd.read_parquet(file_path)
                dataframes.append(df)
        logger.info(f"Loaded taxi data with {len(dataframes)} files.")
        return pd.concat(dataframes, ignore_index=True)
    
    def _load_zone_data(self):
        logger.info("Starting to load zone data.")
        """Load the taxi zone lookup data."""
        file_path = os.path.join(self.data_dir, "taxi_zone_lookup.csv")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded zone data with {len(df)} rows.")
        return df
    
    def _join_taxi_and_zone_data(self):
        logger.info("Joining taxi and zone data.")
        """Join the taxi data with the zone lookup data."""
        taxi_df = self._load_taxi_data()
        zone_df = self._load_zone_data()
        taxi_df = taxi_df.merge(zone_df, left_on='PULocationID', right_on='LocationID', how='left')
        taxi_df = taxi_df.merge(zone_df, left_on='DOLocationID', right_on='LocationID', how='left', suffixes=('_pickup', '_dropoff'))
        logger.info(f"Joined data has {len(taxi_df)} rows.")
        return taxi_df
    
    def load_data(self):
        """Load and join the taxi and zone data."""
        # check if data directory is empty or doesn't exist, if so fetch new data
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir) or self.fetch_new_data:
            logger.info("Data directory is empty or fetch_new_data is True. Fetching new data.")
            self._fetch_data_from_nyc_tlc_website()
            sleep(30) # Wait for 30 seconds to ensure data is fetched
        logger.info("Loading and joining taxi and zone data.")
        taxi_df = self._join_taxi_and_zone_data()
        return taxi_df

    def save_sample_for_preprocessor_tests(self, taxi_df, sample_size=1000):
        """Save a sample of the data for testing purposes."""
        taxi_df = taxi_df.copy()
        sample_df = taxi_df.sample(n=sample_size, random_state=42)
        
        # check if processed data directory exists, if not create it
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        sample_file_path = os.path.join(PROCESSED_DATA_DIR, "test_data_for_preprocessor.parquet")
        sample_df.to_parquet(sample_file_path, index=False)
        return sample_file_path
    
    
class DataLoaderRealTime(DataLoader):
    """DataLoader for real-time data fetching."""
    def __init__(self, raw_data_dir, data_urls, fetch_new_data=False):
        super().__init__(raw_data_dir, data_urls, fetch_new_data=False)
        self.zone_data = self._load_zone_data()  # Load zone data once
    
    def _join_taxi_and_zone_data(self, rt_taxi_df):
        """Join the real-time taxi data with the zone lookup data."""
        taxi_df = rt_taxi_df.copy()
        zone_df = self.zone_data.copy()
        taxi_df = taxi_df.merge(zone_df, left_on='PULocationID', right_on='LocationID', how='left')
        taxi_df = taxi_df.merge(zone_df, left_on='DOLocationID', right_on='LocationID', how='left', suffixes=('_pickup', '_dropoff'))
        return taxi_df
    
    def load_data(self, rt_taxi_df):
        """Load and join the real-time taxi data with the zone data."""
        if rt_taxi_df.empty:
            logger.warning("Received empty real-time taxi DataFrame.")
            return pd.DataFrame()
        logger.info("Joining real-time taxi and zone data.")
        taxi_df = self._join_taxi_and_zone_data(rt_taxi_df)
        return taxi_df
    
    

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesses the taxi data removing logical errors, cleaning the data
    and treating the nulls.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method for compatibility with scikit-learn pipelines."""
        return self
    
    def _keep_necessary_columns(self, df):
        """Keep only the columns that are needed for analysis."""
        columns_to_keep = [
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
            'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag',
            'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount',
            'Borough_pickup', 'service_zone_pickup', 'Borough_dropoff',
            'service_zone_dropoff'
        ]
        return df[columns_to_keep]
    
    def _date_time_conversion(self, df):
        """Convert datetime columns to pandas datetime format."""
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        return df

    def _drop_negative_fares(self, df):
        logger.info("Dropping rows with negative fares.")
        """Remove rows with negative fare amounts."""
        before_rows = len(df)
        df = df[df['fare_amount'] >= 0]
        after_rows = len(df)
        logger.info(f"Dropped {before_rows - after_rows} rows with negative fares.")
        return df
    
    def _drop_negative_duration_trips(self, df):
        logger.info("Dropping trips with negative duration.")
        """Remove trips with negative duration."""
        before_rows = len(df)
        df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
        df = df[df['duration'] >= 0]
        after_rows = len(df)
        logger.info(f"Dropped {before_rows - after_rows} trips with negative duration.")
        return df
    
    def _drop_zero_fares(self, df):
        """Remove rows with zero fare amounts where distance is greater than 0."""
        df = df[(df['fare_amount'] > 0) | (df['trip_distance'] == 0)]
        return df
    
    def _drop_zero_distance_trips(self, df):
        """Remove trips with zero distance when fare is greater than 0."""
        df = df[(df['trip_distance'] > 0) | (df['fare_amount'] == 0)]
        return df
    
    def _drop_zero_duration_trips(self, df):
        """Remove trips with zero duration when trip distance is greater than 0."""
        df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
        df = df[(df['duration'] > 0) | (df['trip_distance'] == 0)]
        return df
    
    def _remove_zero_distance_duration_fare_trips(self, df):
        """Remove trips with zero distance, duration, and fare."""
        df = df[~((df['trip_distance'] == 0) & (df['duration'] == 0) & (df['fare_amount'] == 0))]
        return df

    def _remove_data_with_oob_pickup_times(self, df):
        """Remove data with pickup times not between 2023-01-01 and 2023-03-31 both inclusive."""
        df = df[df['tpep_pickup_datetime'] <= df['tpep_dropoff_datetime']]
        df = df[(df['tpep_pickup_datetime'] >= pd.Timestamp('2023-01-01')) & (df['tpep_pickup_datetime'] < pd.Timestamp('2023-04-01'))]
        return df
    
    def _handle_nulls_using_data_dictionary(self, df):
        """Handle null values based on the data dictionary."""
        # Assuming the data dictionary specifies how to handle nulls for each column
        df.fillna(NA_DICT, inplace=True)
        return df

    def transform(self, X):
        logger.info("Starting data preprocessing.")
        """Main preprocessing function to clean and prepare the data."""
        taxi_df = X.copy()
        taxi_df = self._keep_necessary_columns(taxi_df)
        taxi_df = self._date_time_conversion(taxi_df)
        taxi_df = self._drop_negative_fares(taxi_df)
        taxi_df = self._drop_negative_duration_trips(taxi_df)
        taxi_df = self._drop_zero_fares(taxi_df)
        taxi_df = self._drop_zero_distance_trips(taxi_df)
        taxi_df = self._drop_zero_duration_trips(taxi_df)
        taxi_df = self._remove_data_with_oob_pickup_times(taxi_df)
        taxi_df = self._remove_zero_distance_duration_fare_trips(taxi_df)
        taxi_df = self._handle_nulls_using_data_dictionary(taxi_df)
        logger.info("Data preprocessing completed.")
        return taxi_df
    
    def save_sample_for_feature_engineering_tests(self, taxi_df, sample_size=1000):
        """Save a sample of the data for testing purposes."""
        taxi_df = taxi_df.copy()
        sample_df = taxi_df.sample(n=sample_size, random_state=42)
        
        # check if processed data directory exists, if not create it
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        sample_file_path = os.path.join(PROCESSED_DATA_DIR, "test_data_for_feature_engineering.parquet")
        sample_df.to_parquet(sample_file_path, index=False)
        return sample_file_path
    
    
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering class to create new features from the taxi data."""
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method for compatibility with scikit-learn pipelines."""
        return self

    def _time_features(self, df):
        """Create time-based features from pickup datetime."""
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_minute'] = df['tpep_pickup_datetime'].dt.minute
        df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
        
        # cycling features
        df['pu_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
        df['pu_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
        df['pu_minute_sin'] = np.sin(2 * np.pi * df['pickup_minute'] / 60)
        df['pu_minute_cos'] = np.cos(2 * np.pi * df['pickup_minute'] / 60)
        df['pu_weekday_sin'] = np.sin(2 * np.pi * df['pickup_weekday'] / 7)
        df['pu_weekday_cos'] = np.cos(2 * np.pi * df['pickup_weekday'] / 7)
        
        df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
        
        # drop pickup_hour, pickup_minute, and pickup_weekday columns
        df = df.drop(['pickup_hour', 'pickup_minute', 'pickup_weekday'], axis=1)
        logger.info("Time features created.")
        return df
    
    def _additional_features(self, df):
        # speed
        df['speed'] = df['trip_distance'] / (df['duration'] / 3600)
        
        # route
        df['id_route'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
        df['borough_route'] = df['Borough_pickup'] + '_' + df['Borough_dropoff']
        df['zone_route'] = df['service_zone_pickup'] + '_' + df['service_zone_dropoff']
        
        # pick up and dropoff boroughs and service zones
        df['same_borough'] = (df['Borough_pickup'] == df['Borough_dropoff']).astype(int)
        df['same_service_zone'] = (df['service_zone_pickup'] == df['service_zone_dropoff']).astype(int)
        return df
    
    def transform(self, X):
        logger.info("Starting feature engineering.")
        """Main feature engineering function to create new features."""
        taxi_df = X.copy()
        taxi_df = self._time_features(taxi_df)
        taxi_df = self._additional_features(taxi_df)
        logger.info("Feature engineering completed.")
        return taxi_df
    
    def save_sample_for_null_values_handler_tests(self, taxi_df, sample_size=1000):
        """Save a sample of the data for testing purposes."""
        taxi_df = taxi_df.copy()
        sample_df = taxi_df.sample(n=sample_size, random_state=42)
        
        # check if processed data directory exists, if not create it
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        sample_file_path = os.path.join(PROCESSED_DATA_DIR, "test_data_for_null_values_handler.parquet")
        sample_df.to_parquet(sample_file_path, index=False)
        return sample_file_path
    

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = dict()
            for i, x in enumerate(X[col].unique()):
                le[x] = i
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda x: self.encoders[col].get(x, -1))
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns)
