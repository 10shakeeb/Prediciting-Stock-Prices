import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import style

# Read the datafile, parse dates on "Date"
data = pd.read_csv("sphist.csv", parse_dates=["Date"])

# Sort by "Date" column
data.sort_values("Date", ignore_index=True, inplace=True)

# Display basic info
display(data.info())
display(data.head())

# Initially set the columns to zero
data["prev5_avg"] = 0
data["prev5_std"] = 0
data["prev30_avg"] = 0
data["prev30_std"] = 0
data["prev365_avg"] = 0
data["prev365_std"] = 0

# Iterate over each row to compute the columns using previous 5, 30 and 365 rows (days)
for index, row in data.iterrows():
    # Subset the previous 5 rows, compute the average and standard deviation of "Close" column, and set the columns' values
    past5 = data.iloc[index-5:index]
    past5_avg = past5["Close"].mean()
    past5_std = past5["Close"].std()
    data.loc[index, "prev5_avg"] = past5_avg
    data.loc[index, "prev5_std"] = past5_std
    # Subset the previous 30 rows, compute the average and standard deviation of "Close" column, and set the columns' values
    past30 = data.iloc[index-30:index]
    past30_avg = past30["Close"].mean()
    past30_std = past30["Close"].std()
    data.loc[index, "prev30_avg"] = past30_avg
    data.loc[index, "prev30_std"] = past30_std
    # Subset the previous 365 rows, compute the average and standard deviation of "Close" column, and set the columns' values
    past365 = data.iloc[index-365:index]
    past365_avg = past365["Close"].mean()
    past365_std = past365["Close"].std()
    data.loc[index, "prev365_avg"] = past365_avg
    data.loc[index, "prev365_std"] = past365_std
    
# Display info
display(data.head(10))

# Slice the dataframe starting on row 366 (iloc position 365) so every row has values in our newly created columns
sliced_data = data.iloc[365:]

# Divide the sliced dataframe into Train and Test dataframes
train = sliced_data[sliced_data["Date"] < datetime(year=2013, month=1, day=1)]
test = sliced_data[sliced_data["Date"] >= datetime(year=2013, month=1, day=1)]

# Display head of Train and Test dataframes
display(train.head())
display(test.head())

# Linear Regression model with only one possible predictor
lr = LinearRegression()
lr.fit(train[["prev5_avg"]], train["Close"])
predictions = lr.predict(test[["prev5_avg"]])
rmse = mean_squared_error(predictions, test["Close"]) ** 1/2

# Plot the model vs actual values
style.use("fivethirtyeight")
plt.figure(figsize=(15,10))
plt.plot(test["Date"], test["Close"])
plt.plot(test["Date"], predictions)
plt.legend(["Actual", "Predicted"])
plt.show()

# Display RMSE value
print("RMSE value:", round(rmse,1))

# Linear Regression model with the 6 possible predictors
lr = LinearRegression()
lr.fit(train[["prev5_avg", "prev5_std", "prev30_avg", "prev30_std", "prev365_avg", "prev365_std"]], train["Close"])
predictions = lr.predict(test[["prev5_avg", "prev5_std", "prev30_avg", "prev30_std", "prev365_avg", "prev365_std"]])
rmse = mean_squared_error(predictions, test["Close"]) ** 1/2

# Plot the model vs actual values
style.use("fivethirtyeight")
plt.figure(figsize=(15,10))
plt.plot(test["Date"], test["Close"])
plt.plot(test["Date"], predictions)
plt.legend(["Actual", "Predicted"])
plt.show()

# Display RMSE value
print("RMSE value:", round(rmse,1))
