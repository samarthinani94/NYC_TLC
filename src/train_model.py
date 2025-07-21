import pandas as pd
import numpy as np
import joblib
import os
import sys
sys.path.append("..")  # Adjust path to include the parent directory
from src.preprocessor import MultiColumnLabelEncoder
from src.utils import train_val_test_split, load_and_preprocess_data
from src.logger import get_logger
from src.config import DATA_URLS, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


# if DIRs do not exist, create them
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logger = get_logger(__name__)

# Load and preprocess data
processed_data = load_and_preprocess_data(raw_data_dir=RAW_DATA_DIR, data_urls=DATA_URLS, fetch_new_data=False)
logger.info("Data loading and preprocessing completed successfully.")

# Split data into train, validation, and test sets
X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(processed_data)
logger.info("Data splitting completed successfully.")

# save the train, validation, and test as pickle files
X_train.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_train.pickle'))
y_train.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_train.pickle'))
X_val.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_val.pickle'))
y_val.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_val.pickle'))
X_test.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_test.pickle'))
y_test.to_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_test.pickle'))
logger.info("Train, validation, and test sets saved successfully.")

# One-hot encode categorical features
logger.info("Encoding categorical features...")
num_cat_to_ohe = ['VendorID', 'RatecodeID', 'payment_type']
obj_cat_to_ohe = ['store_and_fwd_flag', 'Borough_pickup', 'service_zone_pickup', 
                    'Borough_dropoff', 'service_zone_dropoff']
obj_cat_to_labelencode = ['id_route', 'borough_route', 'zone_route']

encoder = ColumnTransformer(
    transformers=[
        ('num_cat', OneHotEncoder(handle_unknown='ignore'), num_cat_to_ohe),
        ('obj_cat', OneHotEncoder(handle_unknown='ignore'), obj_cat_to_ohe),
        ('obj_label', MultiColumnLabelEncoder(columns=obj_cat_to_labelencode), obj_cat_to_labelencode)
    ],
    remainder='passthrough'
)

# fit the encoder on the training data
logger.info("Fitting the encoder on the training data...")
encoder.fit(X_train)

# save the encoder for later use
joblib.dump(encoder, os.path.join(MODEL_DIR, 'encoder.pickle'))
logger.info("Encoder saved successfully.")

# Transform the train, validation, and test sets
X_train_encoded = encoder.transform(X_train)
X_val_encoded = encoder.transform(X_val)

try:
    feature_names = encoder.get_feature_names_out()
except:
    feature_names = [f"f_{i}" for i in range(X_train_encoded.shape[1])]
    logger.warning("Could not extract feature names. Using generic names.")

X_train = pd.DataFrame(X_train_encoded, columns=feature_names)
X_val = pd.DataFrame(X_val_encoded, columns=feature_names)

# Manually implementing random grid search for hyperparameter tuning
logger.info("Starting hyperparameter tuning...")
param_grid = {
    'n_estimators': [40, 50],  # set a smaller range for quicker testing
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5],
    'random_state': [42]
}

# create param df with all combinations
param_df = pd.DataFrame([dict(zip(param_grid, v)) for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))])

# 5 random combinations[can be adjusted]; set to 5 for quicker testing
param_df = param_df.sample(n=5, random_state=42).reset_index(drop=True)

# for each combination, train the model and save the train and validation scores
best_train_score = float('inf')
best_val_score = float('inf')
for i, params in param_df.iterrows():
    logger.info(f"Training model with parameters: {params.to_dict()}")
    
    model = XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        max_depth=int(params['max_depth']),
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        reg_lambda=params['reg_lambda'],
        eval_metric='mae',
        early_stopping_rounds=10,
        random_state=int(params['random_state'])
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    # train mae
    train_pred = model.predict(X_train)
    train_score = np.mean(np.abs(train_pred - y_train))
    
    # validation mae
    val_pred = model.predict(X_val)
    val_score = np.mean(np.abs(val_pred - y_val))
    
    logger.info(f"{i}. Train MAE: {train_score}, Validation MAE: {val_score}")

    if train_score < best_train_score:
        best_train_score = train_score

    if val_score < best_val_score:
        best_val_score = val_score
        
        # save the best model
        joblib.dump(model, os.path.join(MODEL_DIR, 'best_hyp_tun_model.pickle'))
        logger.info("New best model saved successfully.")
        
        # log the best parameters found so far
        best_params = params.to_dict()
        logger.info(f"New best parameters: {best_params}")
logger.info("Hyperparameter tuning completed successfully.")

# Final model training with the best parameters    
final_model = XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    max_depth=int(best_params['max_depth']),
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_lambda=best_params['reg_lambda'],
    eval_metric='mae',
    random_state=int(best_params['random_state'])
)

merged_X_train = pd.concat([X_train, X_val], ignore_index=True)
final_y_train = pd.concat([y_train, y_val], ignore_index=True)
final_model.fit(merged_X_train, final_y_train)

# save the final model
joblib.dump(final_model, os.path.join(MODEL_DIR, 'final_model.pickle'))
logger.info("Final model trained and saved successfully.")




