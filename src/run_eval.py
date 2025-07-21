import pandas as pd
import numpy as np
import sys
import os
sys.path.append("..")  # Adjust path to include the parent directory
from src.config import MODEL_DIR, PROCESSED_DATA_DIR, LOG_DIR
from src.utils import adjusted_r2_score, mae_score, median_ae_score, mape_score
import joblib
from tabulate import tabulate
from src.logger import get_logger

logger = get_logger(__name__)

# load the train, validation, and test data
X_train = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_train.pickle'))
y_train = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_train.pickle'))
X_val = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_val.pickle'))
y_val = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_val.pickle'))
X_test = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_test.pickle'))
y_test = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_test.pickle'))
logger.info("Data loaded successfully.")


# load the encoder.pickle file
encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pickle'))

# encode the variables using the encoder
X_train_encoded = encoder.transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)
logger.info("Data encoding completed successfully.")

feature_names = encoder.get_feature_names_out()

X_train = pd.DataFrame(X_train_encoded, columns=feature_names)
X_val = pd.DataFrame(X_val_encoded, columns=feature_names)
X_test = pd.DataFrame(X_test_encoded, columns=feature_names)

# load the best_hyp_tun_model.pickle file
best_hyp_tun_model = joblib.load(os.path.join(MODEL_DIR, 'best_hyp_tun_model.pickle'))
final_model = joblib.load(os.path.join(MODEL_DIR, 'final_model.pickle'))
logger.info("Models loaded successfully.")

# Evaluate the model on the training set
train_adjusted_r2 = adjusted_r2_score(best_hyp_tun_model, X_train, y_train)
train_mae = mae_score(best_hyp_tun_model, X_train, y_train)
train_median_ae = median_ae_score(best_hyp_tun_model, X_train, y_train)
train_mape = mape_score(best_hyp_tun_model, X_train, y_train)
logger.info("Training set evaluation completed.")

# Evaluate the model on the validation set
val_adjusted_r2 = adjusted_r2_score(best_hyp_tun_model, X_val, y_val)
val_mae = mae_score(best_hyp_tun_model, X_val, y_val)
val_median_ae = median_ae_score(best_hyp_tun_model, X_val, y_val)
val_mape = mape_score(best_hyp_tun_model, X_val, y_val)
logger.info("Validation set evaluation completed.")

# Evaluate the model on the test set
test_adjusted_r2 = adjusted_r2_score(best_hyp_tun_model, X_test, y_test)
test_mae = mae_score(best_hyp_tun_model, X_test, y_test)
test_median_ae = median_ae_score(best_hyp_tun_model, X_test, y_test)
test_mape = mape_score(best_hyp_tun_model, X_test, y_test)
logger.info("Test set evaluation completed.")

# Evaluate the model on the test set using final model
test_adjusted_r2_final = adjusted_r2_score(final_model, X_test, y_test)
test_mae_final = mae_score(final_model, X_test, y_test)
test_median_ae_final = median_ae_score(final_model, X_test, y_test)
test_mape_final = mape_score(final_model, X_test, y_test)
logger.info("Test set evaluation completed.")

# feature importance
feature_importance = final_model.feature_importances_
feature_names = encoder.get_feature_names_out()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values(by='importance', ascending=False)

importance_df['feature'] = importance_df['feature'].str.split("__").str[-1]  # Clean up feature names
logger.info("Feature importance calculated successfully.")

# save the evaluation results to a text file
with open(os.path.join(LOG_DIR, 'evaluation_results.txt'), 'w') as f:
    f.write("Evaluation Results using Best Hyperparameter Tuned Model:\n")
    f.write(f"Train Adjusted R2: {train_adjusted_r2}\n")
    f.write(f"Train MAE: {train_mae}\n")
    f.write(f"Train Median AE: {train_median_ae}\n")
    f.write(f"Train MAPE: {train_mape}\n\n")
    
    f.write("Evaluation Results using Best Hyperparameter Tuned Model:\n")
    f.write(f"Validation Adjusted R2: {val_adjusted_r2}\n")
    f.write(f"Validation MAE: {val_mae}\n")
    f.write(f"Validation Median AE: {val_median_ae}\n")
    f.write(f"Validation MAPE: {val_mape}\n\n")
    
    f.write("Evaluation Results using Best Hyperparameter Tuned Model:\n")
    f.write(f"Test Adjusted R2: {test_adjusted_r2}\n")
    f.write(f"Test MAE: {test_mae}\n")
    f.write(f"Test Median AE: {test_median_ae}\n")
    f.write(f"Test MAPE: {test_mape}\n\n")
    
    f.write("Evaluation Results using Final Model:\n")
    f.write(f"Test Adjusted R2 (Final Model): {test_adjusted_r2_final}\n")
    f.write(f"Test MAE (Final Model): {test_mae_final}\n")
    f.write(f"Test Median AE (Final Model): {test_median_ae_final}\n")
    f.write(f"Test MAPE (Final Model): {test_mape_final}\n\n")
    
    f.write("Feature Importance:\n")
    table = tabulate(importance_df.values, headers=importance_df.columns, tablefmt="grid")
    f.write(table)
logger.info("Evaluation results saved successfully.")

