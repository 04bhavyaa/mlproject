# Student Performance Prediction Project

This project aims to predict student math scores based on demographic and educational features using machine learning models. The application is built using Flask and deployed in Docker containers. The project leverages DVC (Data Version Control) for managing datasets and models, and uses CatBoost for model training and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Docker Setup](#docker-setup)
- [Model Training](#model-training)
- [Ongoing Enhancements](#ongoing-enhancements)
- [Contributors](#contributors)

## Project Overview

The objective of this project is to build a machine learning model to predict students' math scores based on features such as gender, race, parental education level, lunch type, and test preparation course status. The project includes:
- A Flask-based web application for real-time predictions.
- A machine learning pipeline using **CatBoost** and **GridSearchCV** for hyperparameter optimization.
- Data and model versioning with **DVC** and **Dagshub**.

## Technologies Used

- **Flask**: For creating the web-based user interface to input data and display predictions.
- **DVC (Data Version Control)**: For managing the data and model versions.
- **Dagshub**: A platform for versioning large datasets and models.
- **Python 3.8+**: The programming language used to implement the machine learning models.
- **CatBoost**: A gradient boosting library used for model training.
- **Docker**: For containerizing the application to ensure consistency across environments.
- **GridSearchCV**: For hyperparameter tuning of the CatBoost model.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Docker (for containerization)
- DVC installed and configured for your cloud storage (Dagshub, AWS, etc.)
- GitHub repository with necessary secrets for Dagshub and Docker Hub

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/mlproject.git
   cd mlproject
2. **Set up a Python environment**:
   ```bash
    conda create --name mlproject python=3.8
    conda activate mlproject
3. **Install dependencies:**

   ```bash
    pip install -r requirements.txt
4. **Set up DVC and pull the data:**
   ```bash
    dvc remote add origin s3://dvc
    dvc pull
5. **Run the Flask application locally:**
   ```bash
    python app.py
The app will be available at http://localhost:5000.

### Running the Application
To run the Flask application locally, follow the steps below:

1. **Activate your environment:**
   ```bash
    conda activate mlproject
2. **Run the Flask app:**
   ```bash
    python app.py
Open the browser and go to http://localhost:5000 to interact with the web application.

### Docker Setup
The application is containerized using Docker. To build and run the app in a Docker container, follow these steps:

1. **Build the Docker image:**
   ```bash
    docker build -t flask-app .
2. **Run the Docker container:**
   ```bash
    docker run -p 5000:5000 flask-app
This will start the Flask app inside a Docker container, and the application will be available at http://localhost:5000.

### Model Training
The machine learning model is trained using various models with hyperparameter tuning performed via GridSearchCV. To train the model, follow these steps:

1. **Train the model by running the model training component:**
   ```bash
    python src/mlproject/component/model_training.py
2. **Evaluate the model using performance metrics (e.g., accuracy, MSE, etc.).**

After training, the model.pkl will be saved in the artifacts/folder, and it will be used for predictions in the Flask app.

### Ongoing Enhancements
- Improve the accuracy of the model by exploring other algorithms and feature engineering techniques.
- Add additional prediction models and compare their performance.
- Enhance the user interface of the web app to include more interactive visualizations.
- Expand the dataset and improve generalization.
- 
### Contributors
Bhavya Jha (Developer)
