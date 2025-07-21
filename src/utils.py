from src.preprocessor import DataLoader, DataPreprocessor, CustomFeatureEngineer
from sklearn.pipeline import Pipeline
from src.config import TRAIN_CUT_OFF_DATE, VAL_CUT_OFF_DATE
from src.logger import get_logger
from sklearn.metrics import mean_absolute_error
import numpy as np

logger = get_logger(__name__)

# Load and preprocess data
def load_and_preprocess_data(raw_data_dir, data_urls, fetch_new_data=False):
    logger.info("Loading data...")
    data_loader = DataLoader(raw_data_dir, data_urls, fetch_new_data=fetch_new_data)
    taxi_df = data_loader.load_data()

    logger.info("Preprocessing and Feature Engineering")
    pfe_pipeline = Pipeline([
        ("preprocessor", DataPreprocessor()),
        ("feature_engineer", CustomFeatureEngineer())
    ])
    processed_data = pfe_pipeline.fit_transform(taxi_df)

    return processed_data


# Split data into training and validation sets
def train_val_test_split(data):
    logger.info("Splitting data into train and validation sets...")
    data = data.copy()
    train_data = data[data['tpep_pickup_datetime'] <= TRAIN_CUT_OFF_DATE]
    val_data = data[(data['tpep_pickup_datetime'] > TRAIN_CUT_OFF_DATE) & 
                    (data['tpep_pickup_datetime'] <= VAL_CUT_OFF_DATE)]
    test_data = data[data['tpep_pickup_datetime'] > VAL_CUT_OFF_DATE]

    X_train, y_train = train_data.drop('fare_amount', axis=1), train_data['fare_amount']
    X_val, y_val = val_data.drop('fare_amount', axis=1), val_data['fare_amount']
    X_test, y_test = test_data.drop('fare_amount', axis=1), test_data['fare_amount']
    
    # drop tpep_pickup_datetime and tpep_dropoff_datetime columns
    X_train = X_train.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
    X_val = X_val.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
    X_test = X_test.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

    logger.info(f"Train set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

# adjusted R2 score calculation
def adjusted_r2_score(model, X, y):
    r2 = model.score(X, y)
    n = X.shape[0]  # number of samples
    p = X.shape[1]  # number of features
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# mae score calculation
def mae_score(model, X, y):
    y_pred = model.predict(X)
    return mean_absolute_error(y, y_pred)

# median absolute error score calculation
def median_ae_score(model, X, y):
    y_pred = model.predict(X)
    return np.median(np.abs(y - y_pred))

# mape score calculation
def mape_score(model, X, y):
    y_pred = model.predict(X)
    # remove zero values from y to avoid division by zero
    non_zero_indices = y != 0
    y = y[non_zero_indices]
    y_pred = y_pred[non_zero_indices]
    if len(y) == 0:
        return np.nan
    return np.mean(np.abs((y - y_pred) / y)) * 100  # MAPE in percentage