```
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('housing_prices.csv')

# Preprocess the data
df = df.drop(['id'], axis=1) # Drop id column
df = pd.get_dummies(df, drop_first=True) # One-hot encode categorical variables
X = df.drop(['price'], axis=1) # Features
y = df['price'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean squared error: {mse:.2f}')
print(f'R-squared value: {r2:.2f}')

# Save the model to disk
joblib.dump(model, 'housing_price_model.pkl')
```
