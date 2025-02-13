import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import joblib


df = pd.read_csv('./data/processed/processed_data.csv')
model = joblib.load('./models/xgboost_model.pkl')

# Split features & label
X = df[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude',
        'Time_Orderd_picked_n', 'day','weekday', 'distance', 'Road_traffic_density', 'Weatherconditions',
       'Driver_age_standardized',  'Delivery_person_Ratings', 'Time_Orderd_n','Vehicle_condition',\
        'City', 'Type_of_vehicle']]            # Features
y = df['target']                            # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Model
model = xgb.XGBRegressor(n_estimators=20, max_depth=9)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
