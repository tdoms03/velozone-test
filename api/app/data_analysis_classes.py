# Import necessary libraries
import pandas as pd
from flask import Flask
from data_analysis import remove_initial_lap, preprocess_lap_times
from faker import Faker
import os


app = Flask(__name__)

MIN_LAP_TIME = 13
MAX_LAP_TIME = 50
TRANS_NAME_FILE = "transponder_names.xlsx"

def load_file(file):
    try:
        df = pd.read_csv(file, delimiter=',', on_bad_lines='skip')
    except:
        df = pd.read_csv(file, delimiter=';', on_bad_lines='skip')
    return df

def remove_outliers(df: pd.DataFrame):
    Q1 = df['lapTime'].quantile(0.25)
    Q3 = df['lapTime'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['lapTime'] >= (Q1 - 1.5 * IQR)) & (df['lapTime'] <= (Q3 + 1.5 * IQR))]
    return df

def generate_random_name():
    """Generate a random name."""
    fake = Faker()
    return fake.first_name()

def load_transponder_names(transponder_ids):
    """Ensure each transponder has a name and save it to an Excel file."""
    if os.path.exists(TRANS_NAME_FILE):
        name_df = pd.read_excel(TRANS_NAME_FILE, dtype={'transponder_id': str})
    else:
        name_df = pd.DataFrame(columns=['transponder_id', 'name'])
    
    # Convert to set for fast lookup
    existing_ids = set(name_df['transponder_id'].astype(str))
    print(existing_ids)
    # Ensure all transponder_ids have a name
    new_entries = [{'transponder_id': str(tid), 'name': generate_random_name()} 
                   for tid in transponder_ids if str(tid) not in existing_ids]

    if new_entries:
        new_df = pd.DataFrame(new_entries)
        name_df = pd.concat([name_df, new_df], ignore_index=True).drop_duplicates(subset=['transponder_id'])
        # Save the updated file
        name_df.to_excel(TRANS_NAME_FILE, index=False)
    return name_df

class DataAnalysis:
    def __init__(self, new_file, debug=True):
        self.file = pd.read_csv(new_file,nrows = 1000)     # Working fine
        self.newlines = self.file.copy()    
        
        # if debug:
        #     print(self.file.head())
        self.debug = debug
        self.cleanup()

        # Load or create transponder name mappings
        transponder_ids = self.file['transponder_id'].unique()
        self.transponder_names = load_transponder_names(transponder_ids)
        # Dataframes that are used to store the important parameters for the screen
        self.average_lap = pd.DataFrame()
        self.fastest_lap = pd.DataFrame()
        self.slowest_lap = pd.DataFrame()
        self.badman = pd.DataFrame()
        self.diesel = pd.DataFrame()
        self.electric = pd.DataFrame()

        # Call the functions that calculate the important parameters for the screen
        self.average_lap_time()
        self.fastest_lap_time()
        self.find_badman()
        self.diesel_engine()
        self.electric_motor()

    def cleanup(self):
        # Convert timestamps to datetime
        self.file['utcTimestamp'] = pd.to_numeric(self.file['utcTimestamp'], errors='coerce')
        self.file.drop_duplicates(inplace = True)
        self.file.dropna(subset=['transponder_id', 'loop', 'utcTimestamp'], inplace=True)
        self.file = preprocess_lap_times(self.file)
        self.file = self.file.sort_values(by=['transponder_id','utcTime'])

        # self.newlines['utcTimestamp'] = pd.to_numeric(self.newlines['utcTimestamp'], errors='coerce')
        # self.newlines.drop_duplicates(inplace = True)
        # self.newlines.dropna(subset=['transponder_id', 'loop', 'utcTimestamp'], inplace=True)
        # self.newlines = preprocess_lap_times(self.newlines)
        # self.newlines = self.newlines.sort_values(by=['transponder_id','utcTime'])
        if self.debug:
            print('cleanup done')

    def update(self, changed_file):
        # Contains the new datarows

        changed_file_pd = changed_file
        # Identify new transponders that were not in the original dataset
        existing_transponders = set(self.transponder_names['transponder_id'].astype(str))  
        new_transponders = set(changed_file_pd['transponder_id']).difference(set(existing_transponders))
        
        if new_transponders:
            print(f"New transponders detected: {new_transponders}")
            new_entries = [{'transponder_id': tid, 'name': generate_random_name()} for tid in new_transponders]

            # Append new transponders to the transponder name mapping
            new_names_df = pd.DataFrame(new_entries)
            self.transponder_names = pd.concat([self.transponder_names, new_names_df], ignore_index=True)

            # Save the updated transponder name list
            self.transponder_names.to_excel(TRANS_NAME_FILE, index=False)

        # Concatenate the new datarows with the existing data, and drop duplicates based on transponder_id and utcTimestamp
        self.file = pd.concat([self.file, changed_file_pd]).drop_duplicates(subset=['transponder_id', 'utcTimestamp'], keep='last')
        self.newlines = pd.merge(changed_file_pd, self.file, how='outer', indicator=True, on=['transponder_id', 'utcTimestamp']).loc[lambda x : x['_merge']=='left_only']
        self.cleanup()  
        
        # call all functions that need to be updated
        self.average_lap_time()
        self.fastest_lap_time()
        self.find_badman()
        self.diesel_engine()
        self.electric_motor()
        if self.debug:
            print('update done')
            print('------------')
    
            
    def average_lap_time(self):
        """
        Function that calculates the average laptime of all the transponders

    Parameters:
        file (str): The file path of the CSV recording context data

    Returns:
        DataFrame: A DataFrame containing the transponder IDs and their respective average lap times

        """
        
        df_sorted = self.file.loc[self.file['loop'] == 'L01']
        avg_lap = df_sorted.groupby('transponder_id')['lapTime'].mean().reset_index()
        
        # Merge with names
        self.average_lap = avg_lap.merge(self.transponder_names, on='transponder_id')
        self.average_lap = self.average_lap[['transponder_id', 'name', 'lapTime']].sort_values(by='lapTime')
        self.average_lap.columns = ['transponder_id', 'name', 'average_lap_time']

        if self.debug:
            print("average_lap_time done.")
    
    def fastest_lap_time(self):
        """
        Function that calculates the fastest lap time for each transponder

        Parameters:
            file (str): The file path of the CSV recording context data
        
        Returns:
            DataFrame: A DataFrame containing the transponder IDs and their respective fastest lap times
        """

        # Sort the data by TransponderID for better visualization and analysis
        df_sorted = self.file.loc[self.file['loop'] == 'L01']

        # Optionally, remove outliers based on IQR or other method
        df_sorted = remove_outliers(df_sorted)
        
        # Calculate the fastest lap time for each transponder ID
        self.fastest_lap = df_sorted.groupby('transponder_id')['lapTime'].min().reset_index()
        # Merge with names
        self.fastest_lap = self.fastest_lap.merge(self.transponder_names, on='transponder_id', how = 'inner')
        self.fastest_lap.columns = ['transponder_id', 'fastest_lap_time', 'name']
        if self.debug:
            print("fastest_lap_time done.")

    def find_badman(self):
        """
        Function that calculates the worst lap time for each transponder

        Returns:
            slowest_lap_time (DataFrame): A DataFrame containing the transponder IDs and their respective worst lap times
            badman (DataFrame): A DataFrame containing the transponder ID and the respective absolute worst lap time of the file
        """
        # Properly filter the DataFrame
        df_sorted = self.file.loc[self.file['loop'] == 'L01']

        # Optionally, remove outliers based on IQR
        df_sorted = remove_outliers(df_sorted)

        # Calculate the slowest lap time for each transponder ID
        self.slowest_lap_time = df_sorted.groupby('transponder_id')['lapTime'].max().reset_index()
        self.slowest_lap_time.columns = ['transponder_id', 'slowest_lap_time']

        # Merge with names
        self.slowest_lap_time = self.slowest_lap_time.merge(self.transponder_names, on='transponder_id', how = 'inner')
        self.slowest_lap_time.columns = ['transponder_id', 'slowest_lap_time', 'name']

        # Construct the BADMAN dataframe
        self.badman = df_sorted.loc[df_sorted['lapTime'].idxmax(), ['transponder_id', 'lapTime']].to_frame().T
        # Merge the worst performer with names
        self.badman = self.badman.merge(self.transponder_names, on='transponder_id', how = 'inner')
        self.badman.columns = ['transponder_id', 'name', 'badman_lap_time']
        if self.debug:
            print("find_badman done.")

    
    def diesel_engine(self,minimum_incalculated = 10,window = 20):
        # Filter only laps recorded at loop 'L01' to focus on complete laps
        df_filtered = self.file.loc[self.file['loop'] == 'L01']
        
        # Drop any rows where 'lapTime' is missing
        df_filtered = df_filtered.dropna(subset=['lapTime'])
        
        # Convert 'lapTime' to numeric values for calculation
        df_filtered.loc[:,'lapTime'] = pd.to_numeric(df_filtered['lapTime'])
        
        # Exclude transponders with fewer than minimum_incalculated laps
        df_filtered = df_filtered.groupby('transponder_id').filter(lambda x: len(x) > minimum_incalculated)
        
        # Calculate the standard deviation (σ) and mean (μ) of lap times for each transponder
        stats = df_filtered.groupby('transponder_id')['lapTime'].agg(['std', 'mean']).reset_index()
        
        # Compute Coefficient of Variation (CV = std / mean), handling potential division by zero
        stats['CV'] = stats['std'] / stats['mean']
        
        # Calculate rolling standard deviation to measure pacing consistency over time
        df_filtered['rolling_std'] = df_filtered.groupby('transponder_id')['lapTime'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        
        # Compute the average rolling standard deviation for each transponder
        rolling_consistency = df_filtered.groupby('transponder_id')['rolling_std'].mean().reset_index()
        rolling_consistency.columns = ['transponder_id', 'rolling_variability']
        
        # Merge consistency metrics
        result = stats.merge(rolling_consistency, on='transponder_id')
        
        # First, select the riders with the lowest rolling variability (most stable pacing)
        most_consistent_riders = result.nsmallest(5, 'rolling_variability')  # Selects top 5 with lowest rolling variability
        
        # Then, from this subset, select the rider with the lowest coefficient of variation (CV)
        self.diesel = most_consistent_riders.nsmallest(1, 'CV')

        # Merge with names
        self.diesel = self.diesel.merge(self.transponder_names, on='transponder_id', how = 'inner')

        if self.debug:
            print("diesel_engine done.")
    
    def electric_motor(self, window=5, lap_distance=250):
        # Filter only laps recorded at loop 'L01' for complete lap measurements
        df_filtered = self.file.loc[self.file['loop'] == 'L01']
        
        # Drop any rows where 'lapTime' is missing
        df_filtered = df_filtered.dropna(subset=['lapTime'])
        
        # Convert 'lapTime' to numeric values for calculation
        df_filtered.loc[:,'lapTime'] = pd.to_numeric(df_filtered['lapTime'])
        
        # Calculate lap speed (assuming lap distance is provided or normalized)
        df_filtered.loc[:,'lapSpeed'] = lap_distance / df_filtered['lapTime']
        
        # Select relevant columns and drop NaN values
        df_filtered = df_filtered[['transponder_id', 'utcTimestamp', 'lapSpeed']].dropna()
        
        # Convert relevant columns to numeric values
        df_filtered['utcTimestamp'] = pd.to_numeric(df_filtered['utcTimestamp'])
        df_filtered['lapSpeed'] = pd.to_numeric(df_filtered['lapSpeed'])
        
        # Sort by transponder and timestamp to ensure correct time sequence
        df_filtered.sort_values(by=['transponder_id', 'utcTimestamp'], inplace=True)
        
        # Calculate speed differences over time
        df_filtered['speed_diff'] = df_filtered.groupby('transponder_id')['lapSpeed'].diff()
        df_filtered['time_diff'] = df_filtered.groupby('transponder_id')['utcTimestamp'].diff()
        
        # Calculate acceleration (change in speed over change in time)
        df_filtered['acceleration'] = df_filtered['speed_diff'] / df_filtered['time_diff']
        
        # Compute rolling maximum acceleration for each transponder
        df_filtered['rolling_acceleration'] = df_filtered.groupby('transponder_id')['acceleration'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
        
        # Find the transponder with the highest peak acceleration
        peak_acceleration = df_filtered.groupby('transponder_id')['rolling_acceleration'].max().reset_index()
        peak_acceleration.columns = ['transponder_id', 'peak_acceleration']
        
        # Identify the transponder with the absolute highest acceleration
        self.electric = peak_acceleration.nlargest(1, 'peak_acceleration')

        if self.debug:
            print("electric_motor done.")


# TODO: 
# 1.  df_filtered can be done centrally and does not need to be repeated