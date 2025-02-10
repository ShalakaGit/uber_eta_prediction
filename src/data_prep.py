from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from preprocessing import *
import pandas as pd


def data_prep(input_data_path, is_train, ordinal_encoders={}, label_encoders={}, scaler=None):
        # Your code goes here
        df = pd.read_csv(input_data_path)
        dataprocess = DataProcessing(df)

        if is_train:
                # Perform Cleaning
                # ## Remove (min) from time_taken
                df['target'] = df['Time_taken(min)'].str.replace('(min)', '', regex=False).astype(int)
                        ## Ordinal encoding for Road_traffice_density, weather, city
                ordinal_encoders = {}
                ordinal_encoders[ 'Road_traffic_density'] = dataprocess.ordinal_encoding(df, 'Road_traffic_density', ['NaN ','Low ', 'Medium ', 'High ', 'Jam '])
                ordinal_encoders[ 'Weatherconditions'] = dataprocess.ordinal_encoding(df, 'Weatherconditions', \
                                                ['conditions NaN','conditions Sunny', 'conditions Windy','conditions Cloudy',
                                                        'conditions Stormy', 
                                                        'conditions Sandstorms',
                                                        'conditions Fog' ])
                ordinal_encoders[ 'City'] = dataprocess.ordinal_encoding(df, \
                                                        'City', ['NaN ', 'Semi-Urban ', 'Urban ', 'Metropolitian '])

                label_encoders = {}
                ## Label encoding type_of_vehicle
                label_encoders['Type_of_vehicle'] = dataprocess.label_encoding(df, 'Type_of_vehicle')

                scaler = StandardScaler()
                df['Driver_age_standardized'] = scaler.fit_transform(df[['Delivery_person_Age']])
        

        ## Modify datatypes 
        c_dtypes = {'Delivery_person_ID': str,
                'ID':str,
                'Delivery_person_Ratings': float}
        for k, v in c_dtypes.items():
                df[k]=df[k].astype(v)
        
        ## Convert NaN values
        dataprocess.convert_nan(df,'Time_Orderd', 'NaN ')
        dataprocess.convert_nan(df,'Time_Order_picked', 'NaN ')

        # ## Get order time for time of the day value
        df['Time_Orderd_n'] = dataprocess.normalize_time_values('Time_Orderd')
        df['Time_Orderd_picked_n'] = dataprocess.normalize_time_values( 'Time_Order_picked')

        ## Compute distance
        df['distance'] = df.apply(lambda row: dataprocess.calculate_haversine_distance(row['Restaurant_latitude'], row['Restaurant_longitude'], 
                                                        row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)

        # Replace null values with the mean
        dataprocess.handle_null_values(df, 'Delivery_person_Ratings')
        df.loc[df['Delivery_person_Age'] == 'NaN ', 'Delivery_person_Age'] = None
        df['Delivery_person_Age'] = dataprocess.update_datatype(df, 'Delivery_person_Age', float)
        dataprocess.handle_null_values(df, 'Delivery_person_Age')

        df['is_festival'] = np.where(df['Festival'] == 'Yes', 1, 0)

        dataprocess.extract_date_features(df, 'Time_Orderd')

        if is_train:
                return df, scaler, label_encoders, ordinal_encoders
        return df
        