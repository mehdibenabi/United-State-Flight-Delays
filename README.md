# United States Flight Delays - Machine Learning Classification

A comprehensive machine learning project that predicts flight delays using PySpark's streaming capabilities. This project simulates real time data streaming to train and evaluate multiple classification models on US flight delay data.

## Project Overview

This project implements a **streaming machine learning pipeline** using PySpark to predict whether a flight will be delayed (≥15 minutes arrival delay). The pipeline processes flight data in batches (chunks) to mimic real time streaming scenarios, trains 5 different classifiers, and evaluates their performance across multiple metrics.

### Key Features
- Real world US flight delay dataset available in Kaggle: ([Dataset](https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024)) 
- Streaming data pipeline using PySpark Structured Streaming
- 5 Machine Learning classifiers trained and evaluated
- Comprehensive model comparison across multiple metrics
- Batch-wise performance tracking and visualization
- Proper data preprocessing with categorical encoding

---

## Project Structure

```
United-State-Flight-Delays/
├── flight_data_2024_sample.csv       # Main dataset (10,000 flight records)
├── flight-delay-notebook-classification.ipynb  # Main Jupyter notebook
├── streaming_input/                  # Input chunks for streaming (created during preprocessing)
│   ├── chunk_0.csv
│   └── ... (generated chunks)
├── streaming/                        # Main data folder provider (for stream processing, Move the csv file to streaming_input folder to mimic real data arrival)
│   ├── chunk_1.csv
│   ├── chunk_2.csv
│   ├── chunk_3.csv
│   └── chunk_4.csv
├── README.md                         
```

---

## Dataset Description

### Source Data: `flight_data_2024_sample.csv`
- **Records**: 10,000 flight records
- **Time Period**: 2024 (US domestic flights)
- **Key Columns**:
  - Flight identifiers: `op_unique_carrier`, `origin`, `dest`
  - Timing: `dep_time`, `arr_time`, `crs_dep_time`, `crs_arr_time`
  - Delays: `dep_delay`, `arr_delay` (in minutes)
  - Flight characteristics: `distance`, `day_of_week`, `month`
  - Cities/States: `origin_city_name`, `dest_city_name`

### Target Variable
- **IsDelayed**: Binary classification
  - `1` = Arrival delay ≥ 15 minutes (delayed)
  - `0` = Arrival delay < 15 minutes (on time)

---

## Technical Stack

- **Apache Spark**: Distributed processing framework for streaming and ML
- **PySpark**: Python API for Spark
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Label encoding for categorical features
- **Matplotlib**: Data visualization
- **Jupyter Notebook**: Interactive development environment

---

## Workflow & Implementation

### 1. **Data Loading & Exploration**
```
- Load flight_data_2024_sample.csv into pandas
- Display dataset shape: (10,000 rows)
- Compute descriptive statistics
- Analyze average delays by month and busiest airports
```

### 2. **Feature Engineering**
Selected 10 key features for modeling:
- `year`, `month` (temporal features)
- `op_unique_carrier` (airline)
- `origin_city_name`, `dest_city_name` (route)
- `dep_time`, `dep_delay` (departure info)
- `arr_time`, `arr_delay` (arrival info, here arr_time was considered as expected arrival time and not real arrival time)
- `distance` (flight distance)

### 3. **Data Preprocessing**
- **Missing Value Handling**: Removed rows with NaN values
- **Categorical Encoding**: Used LabelEncoder to convert categorical features:
  - `op_unique_carrier` → `op_unique_carrier_code`
  - `origin_city_name` → `origin_city_name_code`
  - `dest_city_name` → `dest_city_name_code`
- **Encoded Dataset**: Clean, numeric only data ready for ML models.

### 4. **Data Streaming Preparation**
Split the encoded dataset into **5 equal chunks** (batches):
- Chunk size: 2,000 rows each
- Output: `streaming_input/chunk_0.csv` through `chunk_4.csv`
- Purpose: Simulate continuous data stream from real time sources (streaming folder)

### 5. **PySpark Streaming Pipeline**
- **Schema Definition**: Structured schema for streaming dataframe
- **Stream Source**: Read CSV files from `streaming_input/` directory
- **Foreachbatch Processing**: For each batch of data:
  1. Create binary `IsDelayed` target variable
  2. Remove rows with missing values
  3. Create feature vectors using VectorAssembler
  4. Split data: 80% training, 20% testing
  5. Train all 5 classifiers
  6. Evaluate on test set

### 6. **Machine Learning Models**

Five classifiers were trained and evaluated on each batch:

| Model | Type | Configuration |
|-------|------|----------------|
| **Decision Tree** | Tree based | maxDepth=5 |
| **Random Forest** | Ensemble | numTrees=50, maxDepth=5 |
| **Gradient Boosted Trees (GBT)** | Ensemble | maxIter=20, maxDepth=5 |
| **Linear SVC** | Linear Classifier | maxIter=10 |
| **Logistic Regression** | Linear Classifier | maxIter=20 |

### 7. **Model Evaluation**

Each model is evaluated with **5 comprehensive metrics**:

1. **Accuracy**: Proportion of correct predictions
2. **F1 Score**: Harmonic mean of precision and recall
3. **Precision**: True positives / (true positives + false positives)
4. **Recall**: True positives / (true positives + false negatives)
5. **AUC ROC**: Area under the Receiver Operating Characteristic curve

### 8. **Results Tracking & Visualization**

- **History Tracking**: All metrics recorded for each model across all batches
- **Batch wise Plots**: Track metric evolution over streaming batches
- **Model Comparison**: Average metrics per model across all batches

---

## Key Results

The project generates comprehensive performance metrics showing:
- How each model performs across different batches
- Average performance comparison between all 5 classifiers
- Metric stability/variability across streaming batches
- Best-performing models for flight delay prediction

Example output includes tables with metrics like:
```
--- Decision Tree ---
  Accuracy:  0.9245
  F1:        0.8934
  Precision: 0.8892
  Recall:    0.9273
  AUC:       0.9442
```

---

## Insights & Applications

### Model Strengths
- **Ensemble methods** (Random Forest, GBT) typically outperform Single Decision Tree 
- **Non linear relationships** between features and delays are captured effectively
- **Batch wise evaluation** provides realistic performance expectations

### Use Cases
- Predict flight delays for booking systems
- Airline resource planning and scheduling
- Customer notification systems
- Operations optimization

### Future Enhancements
- Add more features (weather data, historical patterns)
- Implement hyperparameter tuning
- Use multi class classification (early/on time/delayed)
- Deploy as real time streaming service
- Add model persistence and serving capabilities

---

## Running the Project

### Prerequisites
```bash
pip install pyspark pandas matplotlib scikit-learn
```

### Execution Steps

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook flight-delay-notebook-classification.ipynb
   ```

2. **Run cells in order**:
   - Data loading and exploration
   - Feature engineering
   - Data preprocessing
   - Streaming data preparation
   - PySpark initialization
   - Model training and evaluation
   - Visualization generation

3. **Output**: Charts and metrics printed to console

---

## Performance Metrics Explained

### Accuracy
- Overall correctness of predictions
- Equal weight to all classes
- Can be misleading with imbalanced data

### F1 Score
- Balanced metric between precision and recall
- Useful for imbalanced classification
- Harmonic mean: 2 × (precision × recall) / (precision + recall)

### Precision
- How many predicted delays are actually delays
- Important to minimize false alarms
- Precision = TP / (TP + FP)

### Recall
- How many actual delays are caught
- Important to minimize missed delays
- Recall = TP / (TP + FN)

### AUC 
- Measures model's ability to distinguish between classes
- Robust to class imbalance
- Range: 0 to 1 (higher is better)

---

## Code Architecture

### Main Components

1. **Data Processing**: Pandas based cleaning and encoding
2. **Streaming Source**: PySpark CSV stream reader
3. **Feature Engineering**: VectorAssembler for ML pipelines
4. **Model Training**: PySpark ML classifiers
5. **Evaluation**: MulticlassClassificationEvaluator & BinaryClassificationEvaluator
6. **Visualization**: Matplotlib plots with custom styling

### Helper Functions

- `evaluate_model()`: Computes 5 metrics and stores in history dictionary 
- `history_to_df()`: Converts history dict to pandas DataFrame for the plots
- `train_and_evaluate_batch()`: Main batch processing logic
- `plot_metric_over_batches()`: Line plots of metrics over batches
- `plot_avg_metric_bar()`: Bar charts comparing models

---

## Files Generated

| File | Purpose |
|------|---------|
| `streaming_input/chunk_*.csv` | Input batches for streaming |

---

## Author Notes

This project demonstrates:
- End to end ML pipeline with streaming data
- Best practices in data preprocessing
- Proper train & test splitting in distributed environment
- Comprehensive model evaluation
- Reproducible results with fixed seeds (seed=42)

---

## Author

Mehdi Benabi

---

## Contact

For questions or suggestions about this project, feel free to reach out!
