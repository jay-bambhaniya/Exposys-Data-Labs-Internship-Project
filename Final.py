# Importing libraries
import pandas as pd        #Version: 2.2.1
import numpy as np         #Version: 1.26.4
from sklearn.model_selection import train_test_split    #Version: 1.5.0
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the dataset
data = pd.read_csv('50_Startups.csv')

# Data Preprocessing
x = data[['R&D Spend', 'Administration', 'Marketing Spend']].values
y = data['Profit'].values

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)
y_pred_linear = linear_regressor.predict(x_test)

# Polynomial Regression
poly_regressor = PolynomialFeatures(degree=2)
x_poly = poly_regressor.fit_transform(x_train)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(x_poly, y_train)
y_pred_poly = lin_regressor2.predict(poly_regressor.transform(x_test))

# Decision Tree Regression
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(x_train, y_train)
y_pred_tree = tree_regressor.predict(x_test)

# Random Forest Regression
forest_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
forest_regressor.fit(x_train, y_train)
y_pred_forest = forest_regressor.predict(x_test)

# Support Vector Regression
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.transform(x_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).ravel()
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(x_train_scaled, y_train_scaled)
y_pr = svr_regressor.predict(x_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pr.reshape(-1,1)).ravel()

# Calculating Regression Metrics
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

metrics = {
    'Linear Regression': evaluate_model(y_test, y_pred_linear),
    'Polynomial Regression': evaluate_model(y_test, y_pred_poly),
    'Decision Tree Regression': evaluate_model(y_test, y_pred_tree),
    'Random Forest Regression': evaluate_model(y_test, y_pred_forest),
    'Support Vector Regression': evaluate_model(y_test, y_pred_svr)
}

# Selecting the Best Model
best_model = min(metrics, key=lambda x: metrics[x][1]) # Choosing based on lowest MSE

# Displaying Metrics
for model, (mae, mse, r2) in metrics.items():
    print(f"{model}: MAE={mae}, MSE={mse}, R2={r2}")

print(f"Best Model: {best_model}")
