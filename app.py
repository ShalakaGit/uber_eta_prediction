# app.py
import pandas as pd
from src import preprocessing

def main():
    # Your code goes here
    input_data_path = '/Users/shalakathombare/Prepvector course/archive/train.csv'
    df = pd.read_csv(input_data_path)
    dataprocess = preprocessing.DataProcessing()

    # ## Remove (min) from time_taken
    df['target'] = df['Time_taken(min)'].str.replace('(min)', '', regex=False).astype(int)

    c_dtypes = {'Delivery_person_ID': str,
                'ID':str,
                'Delivery_person_Ratings': float}
    for k, v in c_dtypes.items():
        df[k] = dataprocess.update_datatype(df,k, v)
    
    dataprocess.convert_nan(df, 'Time_Orderd', 'NaN ')
    dataprocess.convert_nan(df, 'Time_Order_picked', 'NaN ')


if __name__ == '__main__':
    main()
