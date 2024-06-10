import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import tensorflow as tf
# from tensorflow import keras
# keras = tf.keras
# import tensorflow.python.keras.api._v1.keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed, Dropout, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

class AllPitchDataGenerator(Sequence):
    'Generates data for Keras for all pitches at once'
    def __init__(self, df, n_classes=10, shuffle=False):
        'Initialization'
        self.df = df
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indices = df['PitchUID'].unique()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        'Denotes the number of batches per epoch - in this case, 1 since we are returning everything at once'
        return 1

    def __getitem__(self, index):
        'Generate data for all pitches'
        # Since we're returning all data at once, we ignore 'index' and use all 'indices' instead
        batch_indices = self.indices

        # Initialization
        X, y = [], []

        # Generate data for all pitches
        for pitchUID in batch_indices:
            # Select sequence
            sequence = self.df[self.df['PitchUID'] == pitchUID][['PositionX', 'PositionY', 'PositionZ', 'Time']].values
            # Assuming 'PitchType' is already encoded as integers, adjust as necessary
            label = self.df[self.df['PitchUID'] == pitchUID]['PitchType'].values
            X.append(sequence)
            y.append(to_categorical(label, num_classes=self.n_classes))

        X = pad_sequences(X, padding='post', dtype='float32')
        y = pad_sequences(y, padding='post')

        return X, y

    def on_epoch_end(self):
        'Optionally shuffle indices after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)
class PitchDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=32, n_classes=10, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.df = df
        self.indices = df['PitchUID'].unique()
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indices of the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Initialization
        X, y = [], []

        # Generate data
        for i, pitchUID in enumerate(batch_indices):
            # Select sequence
            sequence = self.df[self.df['PitchUID'] == pitchUID][['PositionX', 'PositionY', 'PositionZ', 'Time']].values
            label = self.df[self.df['PitchUID'] == pitchUID]['PitchType'].values
            X.append(sequence)
            y.append(to_categorical(label, num_classes=self.n_classes))

        X = pad_sequences(X, padding='post', dtype='float32')
        y = pad_sequences(y, padding='post')

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class Driver:
    def __init__(self, db_file, radar_table_name, trajectory_table_name):
        self.db_file = db_file
        self.table_name = radar_table_name
        self.trajectory_table_name = trajectory_table_name
        self.df = []
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.trajectory_df = []

    def read_radar_data (self):
        print ("Reading radar data")
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {self.table_name}', conn).iloc[0, 0]

        # Choose a chunk size
        chunksize = 10000

        # Initialize a progress bar
        pbar = tqdm(total=total_rows)

        # Placeholder DataFrame
        df_list = []

        # Read data in chunks
        for chunk in pd.read_sql_query(f'SELECT * FROM {self.table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])

        # Concatenate all chunks into a single DataFrame
        self.df = pd.concat(df_list, ignore_index=True)
        print (self.df)
        numeric_series = pd.to_numeric(self.df['HitTrajectoryXc8'], errors='coerce')

        # Count the number of non-NaN values (i.e., numeric values)
        num_numeric_values = numeric_series.notna().sum()
        print (num_numeric_values)
        # Close the progress bar
        pbar.close()

        # Close the database connection
        conn.close()

    def read_trajectory_data (self):
        print ("Reading trajectory data")
        conn = sqlite3.connect(f'{self.db_file}')
        total_rows = pd.read_sql_query(f'SELECT COUNT(*) FROM {self.trajectory_table_name}', conn).iloc[0, 0]
        chunksize = 10000
        pbar = tqdm(total=total_rows)
        df_list = []
        for chunk in pd.read_sql_query(f'SELECT * FROM {self.trajectory_table_name}', conn, chunksize=chunksize):
            df_list.append(chunk)
            pbar.update(chunk.shape[0])
        self.trajectory_df = pd.concat(df_list, ignore_index=True)
        print (self.trajectory_df)
        pbar.close()

        # Close the database connection
        conn.close()
    def print_data (self):
        #print ("hi")
        print (self.df)

    def write_data (self):
        db_file = os.path.join(self.current_dir, 'radar2.db')
        conn = sqlite3.connect(db_file)
        self.df.to_sql('radar_data', conn, if_exists='replace', index=False)
        conn.close()

    def find_average_release (self):
        temp_df1 = self.df.groupby('Pitcher')['PitchTrajectoryXc0'].mean().reset_index(name='PitchTrajectoryXc0Average')
        temp_df2 = self.df.groupby('Pitcher')['PitchTrajectoryYc0'].mean().reset_index(name='PitchTrajectoryYc0Average')
        temp_df3 = self.df.groupby('Pitcher')['PitchTrajectoryZc0'].mean().reset_index(name='PitchTrajectoryZc0Average')
        self.df = pd.merge(self.df, temp_df1, on='Pitcher', how='left')
        self.df = pd.merge(self.df, temp_df2, on='Pitcher', how='left')
        self.df = pd.merge(self.df, temp_df3, on='Pitcher', how='left')

    def calculate_trajectories (self):
        time_intervals = np.arange(0, 2, 0.05)

        # Initialize an empty list to store trajectory data for each time interval
        trajectory_data = []
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Computing Trajectoriies..."):
        # For each time interval, calculate the position using the trajectory equation
            for t in time_intervals:
                positionX = (row[f'PitchTrajectoryXc0']) + row[f'PitchTrajectoryXc1'] * t + row[f'PitchTrajectoryXc2'] * t**2
                velocityX = row[f'PitchTrajectoryXc1'] + row[f'PitchTrajectoryXc2'] * t
                positionY = (row[f'PitchTrajectoryYc0']-row[f'PitchTrajectoryYc0Average']) + row[f'PitchTrajectoryYc1'] * t + row[f'PitchTrajectoryYc2'] * t**2
                velocityY = row[f'PitchTrajectoryYc1'] + row[f'PitchTrajectoryYc2'] * t
                positionZ = (row[f'PitchTrajectoryZc0']-row[f'PitchTrajectoryZc0Average']) + row[f'PitchTrajectoryZc1'] * t + row[f'PitchTrajectoryZc2'] * t**2
                velocityZ = row[f'PitchTrajectoryZc1'] + row[f'PitchTrajectoryZc2'] * t
                # Append the calculated position and other relevant information to the trajectory_data list
                if (positionX >= 0):
                    trajectory_data.append({
                        'PitchUID': row['PitchUID'],
                        'Pitcher': row['Pitcher'],
                        'Batter': row['Batter'],
                        'PitchNo': row['PitchNo'],
                        'PitchType':row['AutoPitchType'],
                        'Date': row['Date'],
                        'Time': t,
                        'PositionX': positionX,
                        'VelocityX': velocityX,
                        'PositionY': positionY,
                        'VelocityY': velocityY,
                        'PositionZ': positionZ,
                        'VelocityZ': velocityZ
                    })
                else:
                    break

        # Convert the list of trajectory data into a DataFrame
        trajectory_df = pd.DataFrame(trajectory_data)
        db_file = os.path.join(self.current_dir, 'radar2.db')
        conn = sqlite3.connect(db_file)
        trajectory_df.to_sql('trajectory_data', conn, if_exists='replace', index=False)
        conn.close()

    def build_model (self):
        trajectory_df = self.trajectory_df
        trajectory_df['NewDate'] = pd.to_datetime(trajectory_df['Date'])

        # Filter rows where 'Date' is before 2024
        trajectory_df = trajectory_df[trajectory_df['NewDate'].dt.year >= 2024]

        # Print the number of rows in the filtered DataFrame
        num_rows = len(trajectory_df)
        print(f"Number of rows in the DataFrame after filtering: {num_rows}")
        le = LabelEncoder()
        trajectory_df['PitchType'] = le.fit_transform(trajectory_df['PitchType'])
        num_classes = len(le.classes_)
        pitch_type_to_encoded = dict(zip(le.classes_, range(num_classes)))
        encoded_to_pitch_type = {v: k for k, v in pitch_type_to_encoded.items()}
        print("Original to Encoded Mapping:", pitch_type_to_encoded)
        print("Encoded to Original Mapping:", encoded_to_pitch_type)

# trajectory_df['PitchTypeID'] = pd.factorize(trajectory_df['PitchType'])[0]
        # num_classes = trajectory_df['PitchTypeID'].nunique ()
        unique_pitch_ids = trajectory_df['PitchUID'].unique()
        train_ids, test_ids = train_test_split(unique_pitch_ids, test_size=0.2, random_state=42)

        # Create training and testing DataFrames based on the split IDs
        train_df = trajectory_df[trajectory_df['PitchUID'].isin(train_ids)]
        test_df = trajectory_df[trajectory_df['PitchUID'].isin(test_ids)]
        params = {
            'batch_size': 64,
            'n_classes': num_classes,
            'shuffle': True
        }
        # Creating generators
        train_generator = PitchDataGenerator(df=train_df, **params)
        test_generator = PitchDataGenerator(df=test_df, **params)
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(None, 4)),  # Adjust according to your features
            Dropout(0.5),
            TimeDistributed(Dense(num_classes, activation='softmax'))
        ])
        print ("Compiling")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print ("Fitting")
        # Train the model
        model.fit(train_generator, validation_data=test_generator, epochs=10, verbose=1)
        predictions = model.predict(train_generator)

        # flattened_predictions = predictions.reshape(-1, num_classes)

        full_data_generator = AllPitchDataGenerator(df=trajectory_df, n_classes=num_classes, shuffle=False)
        full_predictions = model.predict(full_data_generator)
        full_predictions = full_predictions.reshape(-1, full_predictions.shape[-1])
        print (full_predictions)
        # predictionsNumpy = full_predictions.numpy ()
        predictions_df = pd.DataFrame (full_predictions, columns = ['Changeup', 'Curveball', 'Cutter', 'Four-Seam', 'Other', 'Sinker', 'Slider', 'Splitter'])
        print(predictions_df)
        merged_df = trajectory_df.merge(predictions_df, left_index=True, right_index=True)
        print (merged_df.to_string ())

        # predictions_df = pd.DataFrame(full_predictions, columns=[f'Pred_Class_{i}' for i in range(full_predictions.shape[1])])
        # print (predictions_df)
        # flattened_predictions = full_predictions.reshape(-1, num_classes)
        # flattened_predictions = tf.reshape(full_predictions.to_tensor(), [-1, num_classes])
        # for i, pitch_type in enumerate(le.classes_):
        #     trajectory_df[f'Prob_{pitch_type}'] = np.nan
        # index = 0
        # for _, group in trajectory_df.groupby('PitchUID'):
        #     length = len(group)  # Number of time steps for the current pitchUID
        #     for i, pitch_type in enumerate(le.classes_):
        #         trajectory_df.loc[trajectory_df['PitchUID'] == group['PitchUID'].iloc[0], f'Prob_{pitch_type}'] = flattened_predictions[index:index+length, i]
        #     index += length
        # print (trajectory_df.to_string ())
        # self.trajectory_df = trajectory_df
        # self.trajectory_df.to_csv("probabilities")
            #
            # # Create sequences
            # sequences = []
            # targets = []
            # for pitchUID, group in trajectory_df.groupby('PitchUID'):
            #     sequences.append(group[['PositionX', 'PositionY', 'PositionZ', 'TimeStep']].values)
            #     targets.append(to_categorical(group['PitchType'], num_classes=num_classes))
            #
            # # Pad sequences to have the same length
            # X = pad_sequences(sequences, padding='post', dtype='float32')
            # y = pad_sequences(targets, padding='post')
            #
            # # Split data (this is a simplified split, consider using sklearn's train_test_split for more control)
            # split_index = int(len(X) * 0.8)
            # X_train, X_test = X[:split_index], X[split_index:]
            # y_train, y_test = y[:split_index], y[split_index:]

            # # Step 2: Build the GRU model
            # model = Sequential([
            #     GRU(64, return_sequences=True, input_shape=(None, X_train.shape[2])),
            #     Dropout(0.5),
            #     TimeDistributed(Dense(num_classes, activation='softmax'))
            # ])
            #
            # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            #
            # # Step 3: Train the model
            # model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
            #
            # # To generate predictions
            # predictions = model.predict(X_test)
driver = Driver ('radar2.db', 'radar_data', 'trajectory_data')
driver.read_trajectory_data()
driver.build_model()
# driver.calculate_trajectories()
# driver.find_average_release()
# driver.print_data()
# driver.write_data()
#driver.print_data()
