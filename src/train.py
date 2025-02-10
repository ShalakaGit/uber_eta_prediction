import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import pickle
from data_prep import data_prep


if __name__ == '__main__':

        # Accepting user input
        input_data_path = input("Enter the file to process: ")
        is_train = input('Do you want to train?, enter True or nothing :')
        
        df, scaler, label_encoders, ordinal_encoders = data_prep(input_data_path, is_train)

        ## Write processed_training data
        df.to_csv('./data/processed/processed_data.csv')

        print('Data written to: ./data/processed/processed_data.csv', )

        # Split features & label
        X = df[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude',\
                'Time_Orderd_picked_n', 'day','weekday', 'distance', 'Road_traffic_density', 'Weatherconditions',\
                'Driver_age_standardized',  'Delivery_person_Ratings', 'Time_Orderd_n','Vehicle_condition',\
                'City', 'Type_of_vehicle', 'is_festival']] # Features
        y = df['target']                            # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build Model
        model = xgb.XGBRegressor(n_estimators = 20, max_depth = 9)
        model.fit(X_train, y_train)

        # Evaluate Model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))

        # Create model.pkl and Save Model
        with open("./models/model.pkl", 'wb') as f:
                pickle.dump((model, label_encoders, scaler, ordinal_encoders), f)

        print("Model has been saved successfully!")
