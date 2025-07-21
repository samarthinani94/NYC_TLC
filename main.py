from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.logger import get_logger
from src.preprocessor import DataLoaderRealTime, DataPreprocessor, CustomFeatureEngineer
from src.config import MODEL_DIR, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_URLS
from sklearn.pipeline import Pipeline
from typing import Optional

app = FastAPI()
logger = get_logger(__name__)

@app.on_event("startup")
def load_model():
    global encoder, model, feature_names
    logger.info("Loading model and encoder...")
    encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pickle'))
    model = joblib.load(os.path.join(MODEL_DIR, 'final_model.pickle'))
    feature_names = encoder.get_feature_names_out()

pre_processing_pipeline = Pipeline([
        ('preprocessor', DataPreprocessor()),
        ('feature_engineer', CustomFeatureEngineer())
])

dlrt = DataLoaderRealTime(raw_data_dir=RAW_DATA_DIR, data_urls=DATA_URLS, fetch_new_data=False)

# request schema for prediction
class TripRequest(BaseModel):
    VendorID: int
    tpep_pickup_datetime: str
    tpep_dropoff_datetime: str
    passenger_count: Optional[int] = None
    trip_distance: float
    RatecodeID: Optional[int] = None
    store_and_fwd_flag: Optional[str] = None
    PULocationID: int
    DOLocationID: int
    payment_type: int
    fare_amount: float

@app.post("/predict")
def predict_fare(data: TripRequest):
    try:
        logger.info("Received prediction request.")
        rt_taxi_df = pd.DataFrame([data.dict()])
        logger.info("Data received for prediction: %s", rt_taxi_df.shape)
        
        # append the zone lookup data
        rt_taxi_df = dlrt.load_data(rt_taxi_df)
        logger.info("Data after loading zone lookup: %s", rt_taxi_df.shape)
        
        # Preprocess the input data
        rt_taxi_df = pre_processing_pipeline.transform(rt_taxi_df)
        logger.info("Data after preprocessing: %s", rt_taxi_df.shape)
        
        # X and y split
        X = rt_taxi_df.drop(columns=['fare_amount'])
        logger.info("Features for prediction: %s", X.shape)
        y = rt_taxi_df['fare_amount']

        X_encoded = encoder.transform(X)
        X = pd.DataFrame(X_encoded, columns=feature_names)
        pred = model.predict(X)
        return {"predicted_fare": round(float(pred[0]), 2), "actual_fare": round(float(y.iloc[0]), 2)}
    except Exception as e:
        logger.exception("Prediction failed.")
        return {"error": str(e)}
    
@app.post("/predict_batch")
def predict_batch_fare(data: list[TripRequest]):
    try:
        logger.info("Received batch prediction request.")
        rt_taxi_df = pd.DataFrame([item.dict() for item in data])
        logger.info("Data received for batch prediction: %s", rt_taxi_df.shape)
        
        # append the zone lookup data
        rt_taxi_df = dlrt.load_data(rt_taxi_df)
        logger.info("Data after loading zone lookup: %s", rt_taxi_df.shape)
        
        # Preprocess the input data
        rt_taxi_df = pre_processing_pipeline.transform(rt_taxi_df)
        logger.info("Data after preprocessing: %s", rt_taxi_df.shape)
        
        # X and y split
        X = rt_taxi_df.drop(columns=['fare_amount'])
        logger.info("Features for batch prediction: %s", X.shape)
        y = rt_taxi_df['fare_amount']

        X_encoded = encoder.transform(X)
        X = pd.DataFrame(X_encoded, columns=feature_names)
        preds = model.predict(X)
        
        return [{"predicted_fare": round(float(pred), 2), "actual_fare": round(float(y_val), 2)} 
                for pred, y_val in zip(preds, y)]
    except Exception as e:
        logger.exception("Batch prediction failed.")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}