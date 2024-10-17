import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import sys


PACKAGE_ROOT =Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))


from prediction_model.config import config

#load the dataset

def load_dataset(file_name):
  filepath = os.path.join(config.DATAPATH,file_name)
  _data = pd.read_csv(filepath)
  _data.columns = [c.strip() for c in _data.columns] # fixes column names
  return _data[config.FEATURES]

# separate X and y
def separate_target(data):
  X = data.drop(config.TARGET, axis =1)
  y = data[config.TARGET]
  return X, y

# split the data
def split_data(X, y, test_size = 0.2, random_state = 0):
  # split into training and testing sets
  X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test

# serialization pipeline 
def save_pipeline(pipeline_to_save):
  save_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_to_save)
  print(save_path)
  joblib.dump(pipeline_to_save, save_path)
  print(f"Model has be saed under the name {config.MODEL_NAME}")

# Deserialization pipeline
def load_pipeline(pipeline_to_load):
  save_path = os.path.join(config.SAVE_MODEL_PATH,
                           pipeline_to_load)
  model_loaded = joblib.load(save_path)
  print(f"Model has be loaded from {save_path}")
  return model_loaded