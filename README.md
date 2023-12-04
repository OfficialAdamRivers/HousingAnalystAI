# HousingAnalystAI
MLA Housing Price Analysis with LRA

This script is a Python script that uses the Scikit-learn library to train a linear regression model on a dataset of housing prices. The script performs the following tasks:

1. Imports necessary libraries, including pandas for data manipulation and Scikit-learn for machine learning.
2. Loads the data from a CSV file named "housing_prices.csv".
3. Drops the "id" column from the data frame and one-hot encodes the categorical variables using the get_dummies() function from pandas.
4. Splits the data into training and testing sets using the train_test_split() function from Scikit-learn.
5. Trains a linear regression model on the training data using the fit() function.
6. Makes predictions on the testing data using the predict() function.
7. Evaluates the model using the mean squared error (MSE) and R-squared values.
8. Saves the trained model to a file named "housing_price_model.pkl" using the joblib.dump() function.

The script assumes that the data is stored in a CSV file named "housing_prices.csv" in the same directory as the script. The script also assumes that the data contains a column named "price" that contains the target variable for the model.
