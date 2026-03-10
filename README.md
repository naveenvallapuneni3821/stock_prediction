📈 Stock Price Prediction using Linear Regression
This project is a simple machine learning application that predicts stock closing prices using a Linear Regression model. It demonstrates how historical stock market data can be used to train a predictive model and estimate future stock prices.
The goal of this project is to understand the complete workflow of a machine learning project, including data preprocessing, feature engineering, model training, evaluation, and visualization.
________________________________________
Project Overview
This project uses Python and Scikit-Learn to build a Linear Regression model that predicts stock closing prices based on historical data.
The program performs several steps during execution:
• Loads stock market data from a CSV file
• Performs feature engineering to create useful prediction variables
• Trains a Linear Regression model using historical data
• Evaluates the model’s performance using standard metrics
• Visualizes the prediction results with graphs
• Predicts the next day’s stock closing price
________________________________________
Technologies Used
Python is the primary programming language used for this project.
Libraries used include:
• Pandas for data handling and preprocessing
• NumPy for numerical computations
• Matplotlib for data visualization
• Scikit-Learn for building and evaluating the machine learning model
________________________________________
Project Workflow
1.	Data Loading
The program begins by loading stock market data from a CSV file. The dataset typically contains fields such as Date, Open, High, Low, Close, and Volume.
If the file is not available, the program automatically generates sample stock data to demonstrate how the model works.
________________________________________
2.	Feature Engineering
To improve the prediction capability of the model, several additional features are created from the existing data. These include:
Yesterday_Close – the previous day's closing price
Price_Change – difference between the open and close price
Daily_Range – difference between the high and low price
MA5 – 5-day moving average of the closing price
Volume – number of shares traded
Open Price – opening price of the stock
These features help the model better understand price trends and patterns.
________________________________________
3.	Model Training
A Linear Regression model from the Scikit-Learn library is trained using 80% of the dataset.
The remaining 20% of the data is reserved for testing the model’s prediction capability.
________________________________________
4.	Model Evaluation
The performance of the trained model is evaluated using two common machine learning metrics:
RMSE (Root Mean Square Error) – measures the average prediction error
R² Score – indicates how well the model explains the variation in stock prices
These metrics help assess how accurately the model predicts stock prices.
________________________________________
5.	Data Visualization
The program generates several visual graphs to help understand the model's performance:
Actual vs Predicted prices for training data
Actual vs Predicted prices for testing data
Scatter plot showing prediction accuracy
Feature importance graph showing which variables influence the model most
________________________________________
6.	Future Price Prediction
After training the model, the program uses the latest available stock data to estimate the next day’s closing price.
This gives an idea of how the trained model can be used for future predictions.
________________________________________
Project Structure
Stock_Prediction_Project
stock_prediction.py – main Python script for data processing, model training, and prediction
stock_data.csv – dataset containing historical stock data
stock_predictions.csv – generated file containing predicted results
README.md – project documentation explaining the workflow and usage

