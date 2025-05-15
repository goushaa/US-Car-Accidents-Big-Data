# Predicting High-Severity Accidents with PySpark MLlib

## Overview
This project focuses on predicting high-severity car accidents (Severity=4) using a binary classification pipeline implemented with PySpark MLlib. The pipeline addresses class imbalance, performs feature engineering, trains machine learning models, evaluates their performance, and interprets key predictors to provide actionable insights for improving road safety.

## Dataset
The dataset used is the **US Accidents Dataset** sourced from Kaggle. It includes details such as:
- **Accident Severity**: Target variable (Severity=4 for high-severity accidents).
- **Location Data**: Latitude, longitude, street, city, state, etc.
- **Weather Conditions**: Temperature, humidity, visibility, wind speed, etc.
- **Road Features**: Presence of junctions, traffic signals, crossings, etc.
- **Time Information**: Start time, end time, and whether the accident occurred during day or night.

**Data Sources**:
- US Accidents Cleaned: [Kaggle Dataset](https://www.kaggle.com/datasets/us-accidents-cleaned)

## Project Structure
The project is divided into three main Jupyter notebooks:
1. **Preprocessing Notebook**:
   - Cleans the dataset by handling missing values and removing redundant columns.
   - Performs data type conversions and prepares the dataset for analysis.
2. **EDA Notebook**:
   - Conducts exploratory data analysis to understand patterns and correlations.
   - Visualizes distributions of key features like severity, weather conditions, and road features.
3. **Model Predictive Notebook**:
   - Implements a binary classification pipeline using PySpark MLlib.
   - Addresses class imbalance through resampling techniques.
   - Engineers features and trains machine learning models.
   - Evaluates model performance using metrics like precision, recall, and F1-score.
   - Interprets key predictors for actionable insights.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.11+
- PySpark 3.x
- Jupyter Notebook
- Required Python libraries: `matplotlib`, `pandas`

You can install the dependencies using:
```bash
pip install pyspark matplotlib pandas
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/goushaa/US-Car-Accidents-Big-Data.git
   cd US-Car-Accidents-Big-Data
   ```

2. **Set Up the Environment**:
   Ensure PySpark and other dependencies are installed as mentioned in the Requirements section.

3. **Run the Notebooks**:
   - Open and run the notebooks in the following order:
     1. `preprocessing_notebook.ipynb`
     2. `eda_notebook.ipynb`
     3. `model_predictive_notebook.ipynb`

4. **View Results**:
   - The preprocessing notebook prepares the dataset.
   - The EDA notebook provides visualizations and insights.
   - The model predictive notebook outputs model performance metrics and key predictors.

## Results
- The project successfully identifies key factors contributing to high-severity accidents, such as weather conditions and road features.
- The trained model achieves balanced performance on the imbalanced dataset through resampling.
- Insights from the model can be used to inform traffic safety policies and interventions.
