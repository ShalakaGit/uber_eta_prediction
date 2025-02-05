import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime

class DataProcessing:
    
    @staticmethod
    def update_column_name(df, old_name, new_name):
        """Update column name in the dataframe."""
        df.rename(columns={old_name: new_name}, inplace=True)
        return df
    
    @staticmethod
    def extract_features(df, feature_columns):
        """Extract specific feature columns from dataframe."""
        return df[feature_columns]
    
    @staticmethod
    def extract_labels(df, label_column):
        """Extract labels column from dataframe."""
        return df[label_column]
    
    @staticmethod
    def drop_columns(df, columns_to_drop):
        """Drop specified columns from the dataframe."""
        df.drop(columns=columns_to_drop, inplace=True)
        return df
    
    @staticmethod
    def update_datatype(df, column_name, new_dtype):
        """Update the datatype of a column in the dataframe."""
        df[column_name] = df[column_name].astype(new_dtype)
        return df
    
    @staticmethod
    def convert_nan(df, column_name, vc):
        """
        Converts the string 'NaN ' to a float NaN value in the given DataFrame df.
        """
        
        df[column_name].replace(vc, float(np.nan), regex=True, inplace=True)


    @staticmethod
    def handle_null_values(df, strategy='mean', fill_value=None):
        """Handle null values in dataframe based on strategy."""
        if strategy == 'mean':
            df.fillna(df.mean(), inplace=True)
        elif strategy == 'median':
            df.fillna(df.median(), inplace=True)
        elif strategy == 'mode':
            df.fillna(df.mode().iloc[0], inplace=True)
        elif strategy == 'value' and fill_value is not None:
            df.fillna(fill_value, inplace=True)
        elif strategy == 'drop':
            df.dropna(inplace=True)
        return df
    
    @staticmethod
    def extract_date_features(df, date_column):
        """Extract year, month, day, weekday, and hour from a datetime column."""
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['weekday'] = df[date_column].dt.weekday
        df['hour'] = df[date_column].dt.hour
        return df
    
    @staticmethod
    def compute_time_delta(df, start_column, end_column):
        """Compute the time delta between two datetime columns."""
        df[start_column] = pd.to_datetime(df[start_column])
        df[end_column] = pd.to_datetime(df[end_column])
        df['time_delta'] = (df[end_column] - df[start_column]).dt.total_seconds()
        return df
    
    @staticmethod
    def deg_to_rad(degrees):
        """Convert degrees to radians."""
        return np.deg2rad(degrees)
    
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two points on Earth."""
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        # Earth radius in kilometers
        radius = 6371.0
        return radius * c
    
    @staticmethod
    def label_encoding(df, column_name):
        """Encode labels in a column using Label Encoding."""
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name])
        return df

    @staticmethod
    def ordinal_encoding_sklearn(df, column_name, categories_order=None):
        """Perform ordinal encoding on a categorical column using sklearn's OrdinalEncoder.
        
        Parameters:
        - df: The pandas DataFrame.
        - column_name: The name of the column to be ordinal encoded.
        - categories_order: A list of categories in the desired order. If None, the encoder will infer the order from the data.
        
        Returns:
        - df: The DataFrame with the ordinal encoded column.
        """
        # Initialize the OrdinalEncoder with the provided categories_order (if specified)
        encoder = OrdinalEncoder(categories=[categories_order] if categories_order else 'auto')
        
        # Fit and transform the specified column
        df[column_name] = encoder.fit_transform(df[[column_name]])
        
        return df

    def data_split(self, X, y):        
        """
        Splits the input features X and target variable y into training and testing sets:
        - Splits the data with a test size of 0.2 and a random state of 42
        - Returns X_train, X_test, y_train, y_test
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def standardize(self, X_train, X_test):
        """
        Standardizes the training and testing feature sets:
        - Fits a StandardScaler on X_train
        - Transforms X_train and X_test using the fitted StandardScaler
        - Returns X_train, X_test, and the fitted StandardScaler
        """
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, scaler


    def cleaning_steps(self, df):
        self.update_column_name(df)
        # self.extract_feature_value(df)
        self.drop_columns(df)
        self.update_datatype(df)
        self.convert_nan(df)
        self.handle_null_values(df)

    def perform_feature_engineering(self, df):
        self.extract_date_features(df)
        self.calculate_time_diff(df)
        self.calculate_distance(df)

    def evaluate_model(self, y_test, y_pred):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print("Mean Absolute Error (MAE):", round(mae, 2))
        print("Mean Squared Error (MSE):", round(mse, 2))
        print("Root Mean Squared Error (RMSE):", round(rmse, 2))
        print("R-squared (R2) Score:", round(r2, 2))