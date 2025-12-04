import yaml,os,json,pickle,mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data_interim(path: str) -> pd.DataFrame:
    """Load processed data from CSV files with debug logging."""
    try:
        print(f"Loading data from {path}")
        df = pd.read_csv(path)
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        df.fillna('', inplace=True)
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return pd.DataFrame()

    
def load_params(path:str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        return {}
    
def load_model(model_path:str):
    """Load the trained model from specified path."""
    try:
        with open(model_path,'rb') as f:
            model=pickle.load(f)
        return model
    except Exception as e:
        return None
    
def load_vectorizer(vectorizer_path:str)->TfidfVectorizer:
    """Load the TF-IDF vectorizer from specified path."""
    try:
        with open(vectorizer_path,'rb') as f:
            vectorizer=pickle.load(f)
        return vectorizer
    except Exception as e:
        return None
    
def load_model_info(path:str)->dict:
    """Load model information including run ID and model path from a JSON file."""
    try:
        with open(path, 'r') as f:
            model_info = json.load(f)
        return model_info
    except Exception as e:
        return {}
    
def load_model_from_registry(model_name,model_version,client):
    """Load model from MLflow Model Registry."""
    try:
        params=load_params("./params.yaml")
        tracking_url = params.get("global", {}).get("tracking_url", "http://ec2-15-222-29-84.ca-central-1.compute.amazonaws.com:5000/")
        mlflow.set_tracking_uri(tracking_url)
        #client = mlflow.tracking.MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        return None