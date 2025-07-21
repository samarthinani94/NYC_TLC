# NYC Taxi Fare Prediction

This project builds a machine learning model to predict the fare amount for a taxi trip in New York City. The model is trained on the yellow taxi trip records from January to March 2023. The final model is served via a FastAPI application.

---

## Deliverables

The output of the project includes:

1. **Trained and Serialized Model**: The trained model will be saved as `final_model.pickle` in the `models/` directory.
2. **Performance Report**: A detailed report on the model's performance will be saved in `logs/evaluation_results.txt`.
3. **Flask API** At the end of Section 3 and 4 
4. `considerations.md`: details regarding decision made for data preprocessing, feature engineering and train-val-test split
5. `todos.md`: Next steps including scope for improvements

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [File Descriptions](#file-descriptions)
   - [`main.py`](#mainpy)
   - [`src/train_model.py`](#srctrain_modelpy)
   - [`src/preprocessor.py`](#srcpreprocessorpy)
   - [`src/utils.py`](#srcutilspy)
   - [`src/run_eval.py`](#srcrun_evalpy)
   - [`src/config.py`](#srcconfigpy)
   - [`src/logger.py`](#srcloggerpy)
   - [`tests/test_preprocess.py`](#test_preprocesspy)
3. [Run with Docker](#run-with-docker)
   - [API Endpoints](#api-endpoints)
4. [How to Run on Local](#how-to-run-on-local)
   - [API Endpoints](#api-endpoints)

---

## Project Structure

```
.
├── main.py             # FastAPI application for serving the model
├── src/
│   ├── config.py         # Project configurations (paths, URLs, etc.)
│   ├── logger.py         # Logging setup
│   ├── preprocessor.py   # Data loading, preprocessing, and feature engineering classes
│   ├── run_eval.py       # Script to evaluate the trained model
│   ├── train_model.py    # Main script to train the model
│   └── utils.py          # Helper functions for data processing and evaluation
├── data/                 # (Created automatically) Stores raw and processed data
├── models/               # (Created automatically) Stores the trained model and encoder
└── logs/                 # (Created automatically) Stores logs and evaluation results
```

---

## File Descriptions

### `main.py`
This script creates a FastAPI application to serve the trained XGBoost model. It loads the model and the preprocessor at startup and provides the following endpoints for fare prediction:

- **`/predict`**: Predicts the fare for a single trip.
- **`/predict_batch`**: Predicts fares for a batch of trips.
- **`/health`**: A health check endpoint.

### `src/train_model.py`
This is the main script for the model training workflow. It performs the following steps:

1. Loads the raw data using functions from `utils.py`.
2. Splits the data into training, validation, and test sets based on dates defined in `config.py`.
3. Defines a `ColumnTransformer` to handle one-hot encoding for categorical features and custom label encoding for high-cardinality features.
4. Fits the encoder on the training data and saves it.
5. Performs a randomized grid search for hyperparameter tuning of an `XGBRegressor` model.
6. Trains the model with the best parameters found and saves the final model to the `models/` directory.

### `src/preprocessor.py`
This file contains all the classes used for data ingestion and transformation:

- **`DataLoader`**: Fetches the raw taxi data from the NYC TLC website and loads it into a pandas DataFrame.
- **`DataPreprocessor`**: Cleans the data by handling logical errors (e.g., negative fares, zero-duration trips) and missing values.
- **`CustomFeatureEngineer`**: Creates new features from the existing data, such as time-based features (hour, weekday) and route-based features.
- **`MultiColumnLabelEncoder`**: A custom transformer for applying label encoding to multiple columns.

### `src/utils.py`
This module contains helper functions used across the project:

- **`load_and_preprocess_data`**: A pipeline function that uses the classes from `preprocessor.py` to load, clean, and feature-engineer the data.
- **`train_val_test_split`**: Splits the data chronologically into training, validation, and test sets.
- **Custom Metrics**: Functions to calculate various evaluation metrics:
  - `adjusted_r2_score`
  - `mae_score`
  - `median_ae_score`
  - `mape_score`

### `src/run_eval.py`
This script is used to evaluate the performance of the trained model. It:

1. Loads the saved model, encoder, and datasets (train, validation, and test).
2. Calculates and prints key performance metrics (Adjusted R², MAE, Median AE, MAPE).
3. Determines and prints the feature importances from the XGBoost model.
4. Saves all evaluation results to `logs/evaluation_results.txt`.

### `src/config.py`
A centralized configuration file that stores all constants and settings:

- Paths to data, model, and log directories.
- URLs for the raw data files.
- A dictionary for handling missing values (`NA_DICT`).
- Cutoff dates for splitting the data into training and validation sets.

### `src/logger.py`
A utility to configure and get a logging object. This ensures that all modules use a consistent logging format, with output directed to both the console and a log file (`logs/output.log`).

### `tests/test_preprocess.py`
This file contains unit tests for the preprocessing pipeline. These tests ensure the correctness of data cleaning, feature engineering, and other preprocessing steps.

---

## Run with Docker

#### Build the Docker Image
NOTE: Currently the data is been manipulated using pandas and measures for memory management haven't been made, so needs more memory. If using Docker Desktop-
1. Open Docker Desktop.
2. Go to Settings > Resources > Advanced.
3. Set Memory Limit to 24GB and Disk Usage limit to 16GB
4. Click Apply & Restart.

##### Single Command to Run - Build, Train and Eval

```bash
docker build -t nyc-taxi-fare . && \
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  nyc-taxi-fare \
  python src/train_model.py && \
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  nyc-taxi-fare \
  python src/run_eval.py
```

- This will download the data, preprocess it, train the model, and save the artifacts (`best_hyp_tun_model.pickle`, `final_model.pickle`, `encoder.pickle`) in the  `models/` directory.
- The results of eval will be saved in `logs/evaluation_results.txt`.

##### Separate Commands

To build the Docker image, run the following command:

```bash
docker build -t nyc-taxi-fare .
```

#### Train the Model
This will download the data, preprocess it, train the model, and save the artifacts (`best_hyp_tun_model.pickle`, `final_model.pickle`, `encoder.pickle`) in the  `models/` directory.

To train the model using Docker, execute the following command:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  nyc-taxi-fare \
  python src/train_model.py
```

#### Evaluate the Model
To evaluate the model, use the following command:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  nyc-taxi-fare \
  python src/run_eval.py
```

The results of eval will be saved in `logs/evaluation_results.txt`

#### Run the FastAPI Application
To start the FastAPI application, run:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  nyc-taxi-fare
```

The API will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## API Endpoints

### `/predict`

- **Method**: `POST`
- **Description**: Predicts the fare for a single taxi trip.
- **Request Body**: A JSON object containing trip details. Example:

```json
{
  "VendorID": 2,
  "tpep_pickup_datetime": "2023-01-01T00:15:00",
  "tpep_dropoff_datetime": "2023-01-01T00:40:00",
  "passenger_count": 1,
  "trip_distance": 4.23,
  "RatecodeID": 1,
  "store_and_fwd_flag": "N",
  "PULocationID": 142,
  "DOLocationID": 236,
  "payment_type": 1,
  "fare_amount": 20.0
}
```

### `/predict_batch`

- **Method**: `POST`
- **Description**: Predicts fares for a batch of taxi trips.
- **Request Body**: A JSON array of objects, each containing trip details. Example:

```json
[
  {
    "VendorID": 2,
    "tpep_pickup_datetime": "2023-01-01T00:15:00",
    "tpep_dropoff_datetime": "2023-01-01T00:40:00",
    "passenger_count": 1,
    "trip_distance": 4.23,
    "RatecodeID": 1,
    "store_and_fwd_flag": "N",
    "PULocationID": 142,
    "DOLocationID": 236,
    "payment_type": 1,
    "fare_amount": 20.0
  },
  {
    "VendorID": 3,
    "tpep_pickup_datetime": "2023-01-01T00:15:00",
    "tpep_dropoff_datetime": "2023-01-01T00:29:00",
    "passenger_count": 3,
    "trip_distance": 3.8,
    "RatecodeID": 1,
    "store_and_fwd_flag": "N",
    "PULocationID": 132,
    "DOLocationID": 123,
    "payment_type": 1,
    "fare_amount": 15.0
  }
]
```

---

## How to Run on Local

### Prerequisites

Ensure you have Python 3.12.3 installed. This project is tested and runs on Python 3.12.3.

### Install Dependencies
Make sure you have the necessary Python libraries installed. You can create a `requirements.txt` file and install them using pip:

```bash
pip install -r requirements.txt
```

Key libraries include:
- `pandas`
- `scikit-learn`
- `xgboost`
- `fastapi`
- `uvicorn`

##### Single Command to Run - Train and Eval
```bash
python -m src.train_model && python -m src.run_eval
```


### Train the Model
Run the training script. This will download the data, preprocess it, train the model, and save the artifacts (`best_hyp_tun_model.pickle`, `final_model.pickle`, `encoder.pickle`) in the `models/` directory:

```bash
python -m src.train_model
```

### Evaluate the Model
After training, run the evaluation script to see the model's performance:

```bash
python -m src.run_eval
```

The results will be saved in `logs/evaluation_results.txt`.

### Run the API Server
Start the FastAPI server to serve the model:

```bash
uvicorn main:app --reload
```

The API will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

