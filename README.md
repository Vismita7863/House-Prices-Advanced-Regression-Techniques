# Advanced Regression for House Price Prediction
This repository contains a complete machine learning project for predicting house prices based on the Ames, Iowa dataset. It covers the entire workflow from data preprocessing and feature engineering to hyperparameter tuning (Optuna), model interpretability (SHAP), and MLOps deployment, moving from exploratory data analysis and hyperparameter tuning to a containerized deployment using Docker, Flask, and Streamlit.

The solution uses a deeply optimized LightGBM model trained on 79 explanatory variables, featuring advanced preprocessing (skew correction, contextual imputation) and interpretability analysis via SHAP.

Repository: https://github.com/Vismita7863/House-Prices-Advanced-Regression-Techniques

## Project Structure

- app/: Contains the production code, including the training pipeline, Flask API, Streamlit frontend, and Docker configuration.
- data/: Contains raw training and testing datasets (AmesHousing).
- experiments/: Contains research scripts, EDA, and historical modeling work (Week 4, 5, 6).
- docs/: Contains project reports, proposals, and presentation slides.
- outputs/: Stores generated artifacts like feature importance plots.

## Prerequisites

- Python 3.8+
- Docker Desktop
- Git

## Installation and Usage

Follow these steps to run the project locally.

### 1. Clone the Repository

Open your terminal and run:

git clone https://github.com/Vismita7863/House-Prices-Advanced-Regression-Techniques.git
cd House-Prices-Advanced-Regression-Techniques

### 2. Environment Setup (Local Python)

If running python scripts locally (without Docker), install the dependencies:

cd app
pip install -r requirements.txt
pip install streamlit

### 3. Train the Model

Before running the API or Docker container, you must generate the model artifact (model_pipeline.pkl). The training script logs experiments to MLflow and saves the pipeline in the current directory.

# Ensure you are inside the 'app' directory
python train_pipeline.py --model lightgbm

Output: This will create 'model_pipeline.pkl' inside the 'app/' folder.

### 4. Option A: Run via Docker (Recommended)

This method runs the API in a container, ensuring consistent behavior across environments. It handles system dependencies (like libgomp1 for LightGBM) automatically.

# Step 1: Build the Docker Image
docker build --no-cache -t ames-housing-api .

# Step 2: Run the Container
# Maps host port 5000 to container port 5000
docker run -p 5000:5000 ames-housing-api

# Step 3: Test the API
# Open a new terminal window, navigate to 'app/', and run:
# Note: Ensure API_URL in test_api.py is set to 'http://127.0.0.1:5000/predict'
python test_api.py

### 5. Option B: Run via Local Flask Server

If you do not wish to use Docker, you can run the Flask API directly.

# Ensure you are inside the 'app' directory
python app.py

# The server will start on http://127.0.0.1:5000

### 6. Run the Frontend (Streamlit)

The project includes a user-friendly web interface to interact with the API.

# Note: Ensure your frontend.py has api_url = "http://127.0.0.1:5000/predict"
# Ensure you are inside the 'app' directory
streamlit run frontend.py

This will open the application in your default web browser at http://localhost:8501.

## Key Technologies

- Modeling: LightGBM, XGBoost, CatBoost, Scikit-Learn
- Tuning: Optuna (Bayesian Optimization)
- Interpretability: SHAP (Shapley Additive exPlanations)
- Tracking: MLflow
- Deployment: Flask (API), Docker (Containerization)
- Frontend: Streamlit

## Authors

- Vaibhav Mahore
- Vismita Tej
