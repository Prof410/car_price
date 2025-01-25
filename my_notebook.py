# The goal of this work is to create a model that will show the highest possible model quality metrics.
# I used CatBoost for machine learning purposes, while other libraries were used minimally to keep the code as simple and clear as possible.
# The model achieved an R2 Score of 0.94.

# Import Modules
import pandas as pd

! pip install catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score

# https://drive.google.com/file/d/1tvWAFIAEWL_HfQURmZRKf2xLy7NCPTrk/view?usp=sharing
! gdown --id 1tvWAFIAEWL_HfQURmZRKf2xLy7NCPTrk
df = pd.read_csv('CarPrice_Assignment.csv')

df.head()

# check the size
df.shape

# Check the number of unique values of each column
df.nunique()

# "car_ID" is the entry number in the data. Since the number of unique records is equal to the total number of records.
# This column needs to be removed.
df.drop('car_ID', axis=1, inplace = True)

# Check the datatype
df.dtypes

# Chesk the columns
df.columns

# X - all features, except 'price'
X = ['symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
     'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype',
     'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# cat_features  - categorical features
cat_features = ['CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel','enginelocation',
                'enginetype', 'cylindernumber', 'fuelsystem']

# y - target variable
y = ['price']

# dividing data into training and test
train, test = train_test_split(df,train_size=0.8, random_state=42)

# setting training parameters
parameters = {'cat_features': cat_features,
              'eval_metric': 'RMSE',
              'random_seed':42,
              'verbose':100}

model = CatBoostRegressor(**parameters)

# model training
model.fit(train[X],train[y])

# prediction on test data
predictions = model.predict(test[X])

# calculation mse and r2:
mse = mean_squared_error(test[y], predictions)
print(f"MSE: {mse}")
r2 = r2_score(test[y], predictions)
print(f"R²: {r2}")

# As a result, the model showed accuracy:
# MSE: 4 875 574.50
# R²: 0.94