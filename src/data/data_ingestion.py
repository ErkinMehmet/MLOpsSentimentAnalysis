import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os,yaml,logging

# logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler('data_ingestion.log')
file_handler.setLevel('ERROR')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(path:str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        logger.debug("Parameters loaded successfully.")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        return {}

def load_data(data_url:str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded successfully from CSV.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()
    
def preprocess_data(df:pd.DataFrame,) -> pd.DataFrame:
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df=df[df['clean_comment'].str.strip()!='']
        return df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return pd.DataFrame()
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,path:str) -> None:
    """Save train and test data to specified path."""
    try:
        raw_path=os.path.join(path,'raw')
        os.makedirs(raw_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_path, 'test.csv'), index=False)
        logger.debug("Train and test data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")

def main():
    try:
        params = load_params("./params.yaml")
        test_size = params.get("data_ingestion", {}).get("test_size", 0.2)
        
        data_url = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
        df = load_data(data_url)
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, "./data")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()