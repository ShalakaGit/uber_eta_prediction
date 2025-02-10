import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


class DataProcessing:
    def __init__(self, df):
        """
        Initializes the DataPreprocessor class with a DataFrame.
        """
        self.df = df

    @staticmethod
    def update_column_name(self, old_name, new_name):
        """Update column name in the dataframe."""
        self.df.rename(columns={old_name: new_name}, inplace=True)
    
    @staticmethod
    def extract_features(self, feature_columns):
        """Extract specific feature columns from dataframe."""
        return self.df[feature_columns]
    
    @staticmethod
    def extract_labels(self, label_column):
        """Extract labels column from dataframe."""
        return self.df[label_column]
    
    @staticmethod
    def drop_columns(self, columns_to_drop):
        """Drop specified columns from the dataframe."""
        self.df.drop(columns=columns_to_drop, inplace=True)
    
    @staticmethod
    def update_datatype(self, column_name, new_dtype):
        """Update the datatype of a column in the dataframe."""
        return self[column_name].astype(new_dtype)
    
    @staticmethod
    def convert_nan(self, column_name, vc):
        """
        Converts the string 'NaN ' to a float NaN value in the given DataFrame df.
        """
        self[column_name].replace(vc, float(np.nan), regex=True, inplace=True)


    @staticmethod
    def handle_null_values(self, col, strategy='mean', fill_value=None):
        """Handle null values in dataframe based on strategy."""
        if strategy == 'mean':
            self[col].fillna(self[col].mean(), inplace=True)
        elif strategy == 'median':
            self[col].fillna(self[col].median(), inplace=True)
        elif strategy == 'mode':
            self[col].fillna(self[col].mode().iloc[0], inplace=True)
        elif strategy == 'value' and fill_value is not None:
            self[col].fillna(fill_value, inplace=True)
        elif strategy == 'drop':
            self[col].dropna(inplace=True)
    
    @staticmethod
    def extract_date_features(self, date_column):
        """Extract year, month, day, weekday, and hour from a datetime column."""
        self[date_column] = pd.to_datetime(self[date_column])
        self['year'] = self[date_column].dt.year
        self['month'] = self[date_column].dt.month
        self['day'] = self[date_column].dt.day
        self['weekday'] = self[date_column].dt.weekday
        self['hour'] = self[date_column].dt.hour
    
    @staticmethod
    def compute_time_delta(self, start_column, end_column):
        """Compute the time delta between two datetime columns."""
        self.df[start_column] = pd.to_datetime(self.df[start_column])
        self.df[end_column] = pd.to_datetime(self.df[end_column])
        self.df['time_delta'] = (self.df[end_column] - self.df[start_column]).dt.total_seconds()
        
    
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
    def label_encoding(self, column_name):
        """Encode labels in a column using Label Encoding."""
        le = LabelEncoder()
        self[column_name] = le.fit_transform(self[column_name])
        return le
        

    @staticmethod
    def ordinal_encoding(self, column_name, categories_order=None):
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
        self[column_name] = encoder.fit_transform(self[[column_name]])
        return encoder

        
    @staticmethod
    def data_split(self):        
        """
        Splits the input features X and target variable y into training and testing sets:
        - Splits the data with a test size of 0.2 and a random state of 42
        - Returns X_train, X_test, y_train, y_test
        """
        # Split features & label
        X = self.drop('target', axis=1)               # Features
        y = self['target']                            # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def standardize(self, df):
        """
        Standardizes the training and testing feature sets:
        - Fits a StandardScaler on X_train
        - Transforms X_train and X_test using the fitted StandardScaler
        - Returns X_train, X_test, and the fitted StandardScaler
        """
        
        # scaler = StandardScaler()
        # scaler.fit(df)
        # X_test = scaler.transform(X_test)
        # return X_train, X_test, scaler
    

    def normalize_time_values(self, col):
        print('normalize time values: ', self.df.sample(), ':', col)
        return (pd.to_datetime(self.df[col]).dt.hour * 3600 +\
                 pd.to_datetime(self.df[col]).dt.minute * 60 + \
                            pd.to_datetime(self.df[col]).dt.second) / 86400


    def cleaning_steps(self):
        pass
        
        

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
