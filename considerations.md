# Assumptions in Data Preprocessing and Feature Engineering

This document outlines the key assumptions made during the data cleaning, preprocessing, and feature engineering phases of the NYC Taxi Fare Prediction project. These decisions are based on domain knowledge and aim to improve data quality and model performance.

---

## 1. Data Preprocessing Assumptions

### Logical Data Cleaning

#### Negative Fares and Durations
- **Action**: Removed trips with negative `fare_amount` or a negative calculated duration.
- **Assumption**: Negative values in these fields are considered data entry errors or records of disputes/refunds, not valid trips. A trip cannot logically have a negative cost or last for a negative amount of time.

#### Zero-Value Trips
- **Action**: Removed trips where:
  - `fare_amount` was zero but `trip_distance` was greater than zero.
  - `trip_distance` was zero but `fare_amount` was positive.
  - Duration was zero but `trip_distance` was positive.
- **Assumption**: These scenarios represent logical inconsistencies. A trip that covers a distance should have a fare and take time. While exceptions (e.g., customer cancellations) exist, these are treated as outliers or data errors that could confuse the model.

#### Zero Distance, Duration, and Fare
- **Action**: Removed records where `trip_distance`, duration, and `fare_amount` were all zero.
- **Assumption**: These records contain no useful information for predicting fare amount and likely represent trips that were initiated but never started, or are simply invalid data points.

### Data Filtering and Scope

#### Time Range
- **Action**: Kept only the records where `tpep_pickup_datetime` was between January 1, 2023, and March 31, 2023.
- **Assumption**: The project's scope is limited to the first quarter of 2023. Any data outside this range is considered out-of-scope.

#### Column Selection
- **Action**: Selected a specific subset of columns for the model. Dropped columns like `tip_amount`, `tolls_amount`, `total_amount` etc.
- **Assumption**: The selected columns are the most relevant predictors for `fare_amount`. Dropping `total_amount` prevents data leakage, as it is calculated from the fare. Other dropped columns are assumed to be either components of the final fare or not directly useful as initial predictors.

### Handling Missing Values

- **Action**: Filled missing values using a predefined dictionary (`NA_DICT`):
  - `passenger_count` → 1: Assumes missing passenger count corresponds to a single rider, the most common scenario.
  - `RatecodeID` → 99: 99 is the code for unknown `RatecodeID`.
  - `store_and_fwd_flag` → 'Z': Assumes missing values for this flag can be treated as their own category.
  - Borough and `service_zone` → 'Outside of NYC' or 'Unknown': Assumes missing location data implies the trip originated or ended outside standard taxi zones, or the information is genuinely unknown. This creates a specific category for such trips.

---

## 2. Feature Engineering

### Time-Based Features

- **Action**: Extracted `hour`, `minute`, and `weekday` from the pickup time and converted them into cyclical features using sine and cosine transformations.
- **Action**: Created an `is_weekend` feature.
- **Assumption**: Fare and traffic patterns differ significantly between weekdays and weekends (Saturday/Sunday). This binary flag helps the model capture that difference.

### Trip-Based Features

#### Average Speed
- **Action**: Calculated the average speed of the trip.
- **Assumption**: The average speed is a proxy for traffic conditions. A lower speed might correlate with a longer duration and potentially a higher fare, making it a valuable predictive feature.

#### Combined Route Features
- **Action**: Created combined route features (e.g., `id_route`, `borough_route`).
- **Assumption**: The specific route, defined by the pickup and dropoff location pair, is a powerful predictor of the fare. Combining them into a single categorical feature allows the model to learn specific fare structures for popular or unique routes.

#### Same Borough and Service Zone Flags
- **Action**: Created `same_borough` and `same_service_zone` flags.
- **Assumption**: Trips that occur entirely within a single borough or service zone may have different fare structures compared to trips that cross boundaries. This feature helps the model capture such localized effects.

---

## 3. Train-Validation-Test Split

### Data Splitting Logic

- **Action**: Split the dataset into three subsets: training, validation, and test sets based on the `tpep_pickup_datetime` column.
  - Training set: Records where `tpep_pickup_datetime` is on or before `2023-03-17` (defined as `TRAIN_CUT_OFF_DATE`).
  - Validation set: Records where `tpep_pickup_datetime` is after `2023-03-17` but on or before `2023-03-24` (defined as `VAL_CUT_OFF_DATE`).
  - Test set: Records where `tpep_pickup_datetime` is after `2023-03-24`.

### Target Variable

- **Action**: The `fare_amount` column is used as the target variable (`y`) for all subsets.
