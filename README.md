# Student Performance Prediction Project

This project aims to predict student math scores based on demographic and educational features using machine learning models. The application is built using Flask and deployed in Docker containers. The project leverages DVC (Data Version Control) for managing datasets and models, and uses CatBoost for model training and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Gallery](#project-galley)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Docker Setup](#docker-setup)
- [Model Training](#model-training)
- [Ongoing Enhancements](#ongoing-enhancements)
- [Contributors](#contributors)

## Project Overview

The objective of this project is to build a machine learning model to predict students' math scores based on features such as gender, race, parental education level, lunch type, and test preparation course status. The project includes:
- A Flask-based web application for real-time predictions.
- Machine learning pipeline and components using various models such as **CatBoost**, **XGBoost**, and **Random Forest** used for predictions and **GridSearchCV** for hyperparameter optimization.
- Data and model versioning with **DVC** and **Dagshub**.

## Technologies Used
- Flask: Built a user-friendly web interface for inputting data and displaying predictions.
- DVC: For managing data and model versioning efficiently.
- Dagshub: To version large datasets and track experiments.
- GitHub Actions: Automated workflows for CI/CD using YAML configurations.
- Python 3.8+: The backbone of the entire project.
- Git: Version-controlled the code and collaborated efficiently.
- VSCode: My go-to code editor for writing, testing, and debugging the project.
- Docker: For containerizing the app and ensuring consistency across environments.
- ML Models: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, and AdaBoost.
- GridSearchCV: To fine-tune the CatBoost model for optimal performance.
- Libraries: NumPy, Pandas, Scikit-learn, XGBoost, CatBoost, Matplotlib, Seaborn, and more.

## Project Gallery
![image](https://github.com/user-attachments/assets/731edf55-cf99-42fd-a9a6-8ceb98d00623)
![image](https://github.com/user-attachments/assets/05a1aebf-0c56-4014-b993-f0084d005311)

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

1. **Train the model by running component.py:**
   ```bash
    python components.py
2. **Evaluate the model using performance metrics (e.g., accuracy, MSE, etc.)**
3. **To edit the component you can go to src/mlproject/components/model_trainer.py**

After training, the model.pkl will be saved in the artifacts/folder, and it will be used for predictions in the Flask app.
Best Model as of now: Lasso regression with the highest Test R2 score of 0.8812.

### Ongoing Enhancements
- Improve the accuracy of the model by exploring other algorithms and feature engineering techniques.
- Add additional prediction models and compare their performance.
- Enhance the user interface of the web app to include more interactive visualizations.
- Expand the dataset and improve generalization.

### Contributors
Bhavya Jha (Developer)
