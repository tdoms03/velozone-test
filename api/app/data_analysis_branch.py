# Import necessary libraries
import pandas as pd
from flask import Flask, render_template
from faker import Faker
import os

app = Flask(__name__)

TRANS_NAME_FILE = "transponder_names.xlsx"
MIN_LAP_TIME = 13
MAX_LAP_TIME = 50

def load_file(file: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, delimiter=',', on_bad_lines='skip')
    except:
        df = pd.read_csv(file, delimiter=';', on_bad_lines='skip')
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

def preprocess_lap_times(df):
    """Operations:
    - Ensure lapTime is numeric
    - Drop rows with NaN in the 'lapTime' column
    - Filter out lap times that are too short or too long"""

    df['lapTime'] = pd.to_numeric(df['lapTime'], errors='coerce')  # Ensure lapTime is numeric
    valid_laps = df[(df['lapTime'] >= MIN_LAP_TIME) & (df['lapTime'] <= MAX_LAP_TIME)]
    valid_laps = valid_laps.dropna(subset=['lapTime']) # Drop rows with NaN in the 'lapTime' column
    return valid_laps

class DataAnalysis:
    def __init__(self, new_file, debug=False):
        columns_incomming_csv = ['transponder_id','loop','utcTimestamp','utcTime','lapTime','lapSpeed','maxSpeed','cameraPreset','cameraPan','cameraTilt','cameraZoom','eventName','recSegmentId','trackedRider']
        self.file = pd.DataFrame(columns=columns_incomming_csv)
        self.newlines = pd.DataFrame(columns=columns_incomming_csv)

        # self.fileL01 = self.file.loc[self.file['loop'] == 'L01']
        # self.newlinesL01 = self.newlines.loc[self.newlines['loop'] == 'L01']

        self.info_per_transponder = pd.DataFrame(columns=['transponder_id', 'transponder_name', 'fastest_lap_time', 'average_lap_time', 'slowest_lap_time'])
        self.newlines_without_outliers = pd.DataFrame()

        self.outliers = pd.DataFrame()

        self.debug = debug

        self.update(new_file)

    def cleanup(self):
        self.file.drop_duplicates(inplace = True)
        self.newlines = self.newlines.dropna(subset=['transponder_id', 'loop', 'utcTimestamp'], inplace=True).sort_values(by=['transponder_id','utcTime'])

    def update(self, changed_file:str):
        """
        Loads the changed lines from the CSV file and appends them to the existing DataFrame.
        Then it updates the following:
        - The transponder names
        - The average lap time for each transponder
        - The fastest lap time for each transponder
        - The badman (the transponder with the highest average lap time)
        - The diesel engine (the transponder with the lowest average lap time)
        - The electric motor (the transponder with the highest average lap time among the electric transponders)

        Parameters:
            changed_file (str): The path to the CSV file with the changed data
        """
        # load the changed lines and append them to the file
        self.newlines = preprocess_lap_times(load_file(changed_file))
        self.file = pd.concat([self.file, self.newlines]).drop_duplicates(subset=['transponder_id', 'utcTimestamp'], keep='last')
        
        self.cleanup()  # TODO: does the whole file needs to be cleaned up/sorted after an update, or is cleanup from the newlines enough?
        
        # update the transponder names if necessary
        existing_transponders = set(self.info_per_transponder['transponder_id'].astype(str))
        new_transponders = set(self.newlines['transponder_id']).difference(set(existing_transponders))


        # call all functions that need to be updated
        self.average_lap_time()
        self.fastest_lap()
        self.badman()
        self.diesel_engine()
        self.electric_motor()

        if self.debug:
            print('update done\n'+'='*40)


    def average_lap_time(self):
        """
        Function that calculates the average laptime of all the transponders

        Returns:
            DataFrame: A DataFrame containing the transponder IDs and their respective average lap times
        """
        # Sort the data by TransponderID for better visualization and analysis
        # df_sorted = self.file.where(self.file['loop'] =='L01').dropna(subset='loop') # I think this isn't needed, as the csv file already contains a column lapTime
        
        # Calculate the average lap time for each transponder ID
        new_average_lap_time = self.fileL01[self.file['transponder_id'].isin(self.newlines['transponder_id'])].groupby('transponder_id')['lapTime'].mean().reset_index()   # only calculate the averages for the updated transponders
        self.info_per_transponder.update(new_average_lap_time.set_index('transponder_id')[['average_lap_time']])

        if self.debug:
            print(self.info_per_transponder.head())

    def remove_outliers(self, outlier_threshold=100):      # TODO: only remove outliers if number of lines of self.file is greater than a certain amount? Which is this amount?
        if len(self.file) < outlier_threshold:
            self.newlines_without_outliers = self.newlines
            return


        Q1 = self.file['lapTime'].quantile(0.25)
        Q3 = self.file['lapTime'].quantile(0.75)
        IQR = Q3 - Q1

        self.newlines_without_outliers = self.newlines[(self.newlines['lapTime'] >= (Q1 - 1.5 * IQR)) & (self.newlines['lapTime'] <= (Q3 + 1.5 * IQR))]
        
    
    def fastest_lap(self):      # self.remove_ouliers should be run before this function is called
        """
        Function that calculates the fastest lap time for each transponder.
        
        Returns:
            DataFrame: A DataFrame containing the transponder IDs and their respective fastest lap times
        """
        # df_sorted = self.file.where(self.file['loop'] =='L01').dropna(subset='loop') # I think this isn't needed, as the csv file already contains a column lapTime
        new_potential_fastest_lap_time = self.newlines_without_outliers.groupby('transponder_id')['lapTime'].min().reset_index()
        
        for index, row in new_potential_fastest_lap_time.iterrows():
            if self.info_per_transponder.loc[self.info_per_transponder['transponder_id'] == row['transponder_id'], 'fastest_lap_time'].values[0] > row['lapTime']:
                self.info_per_transponder.loc[self.info_per_transponder['transponder_id'] == row['transponder_id'], 'fastest_lap_time'] = row['lapTime']
        # TODO: research if it is possible without a for loop (i don't think so as there has to happen a term by term comparison)

    def slowest_lap(self):      # self.remove_ouliers should be run before this function is called
        """
        Function that calculates the slowest lap time for each transponder.
        
        Returns:
            DataFrame: A DataFrame containing the transponder IDs and their respective slowest lap times
        """
        new_potential_slowest_lap_time = self.newlines_without_outliers.groupby('transponder_id')['lapTime'].max().reset_index()

        for index, row in new_potential_slowest_lap_time.iterrows():
            if self.info_per_transponder.loc[self.info_per_transponder['transponder_id'] == row['transponder_id'], 'slowest_lap_time'].values[0] < row['lapTime']:
                self.info_per_transponder.loc[self.info_per_transponder['transponder_id'] == row['transponder_id'], 'slowest_lap_time'] = row['lapTime']

    def badman(self):
        """
            Function that calculates the slowest rider of the session.
            Returns:
                DataFrame: A DataFrame containing the transponder ID and the corresponding slowest lap time.
        """
        slowest_rider = self.info_per_transponder.loc[self.info_per_transponder['slowest_lap_time'].idxmax()]
        return slowest_rider
    
    def diesel_engine(self,minimum_incalculated = 10,window = 20):
        # Filter only laps recorded at loop 'L01' to focus on complete laps
        df_filtered = self.file.loc[self.file['loop'] == 'L01']
        
        # Drop any rows where 'lapTime' is missing
        df_filtered.dropna(subset=['lapTime'], inplace=True)
        
        # Convert 'lapTime' to numeric values for calculation
        df_filtered['lapTime'] = pd.to_numeric(df_filtered['lapTime'])
        
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