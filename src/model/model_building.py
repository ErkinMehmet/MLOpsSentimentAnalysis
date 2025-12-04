import numpy as np
import pandas as pd
import os,pickle,yaml,logging
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from src.utils.logger import get_logger
from src.utils.commonFuncs import get_root_directory
from src.utils.load import load_params,load_data_interim

# logging configuration
logger = get_logger("model_building")
    
def apply_tfidf(train_data:pd.DataFrame,ngram_range:tuple,max_features:int)->tuple: # only fit vectorizer on train data
    try:
        logger.debug(f"Starting TF-IDF with ngram_range={ngram_range}, max_features={max_features}")
        logger.debug(f"Columns in train_data: {train_data.columns.tolist()}")
        logger.debug(f"First 5 rows of 'clean_comment':\n{train_data['clean_comment'].head()}")
        if 'clean_comment' not in train_data.columns or 'category' not in train_data.columns:
            raise KeyError("train_data debe contener las columnas 'clean_comment' y 'category'")

        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        X_train = vectorizer.fit_transform(train_data['clean_comment'].values)
        y_train = train_data['category'].values
        logger.debug(f"TF-IDF shape: {X_train.shape}")
        logger.debug(f"y_train shape: {y_train.shape}")
        logger.debug("TF-IDF transformation applied successfully.")
        # save the vectorizer in the root directory
        vectorizer_path = os.path.join(get_root_directory(),'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug(f"Vectorizer saved successfully at {vectorizer_path}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Error in TF-IDF transformation: {e}")
        return None, None
    
def train_model(X_train:np.ndarray, y_train:np.ndarray, learning_rate:float, max_depth:int, n_estimators:int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',num_class=3,metric='multi_logloss',is_unbalance=True,class_weight='balanced',
            reg_alpha=0.1,reg_lambda=0.1,
            learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        logger.debug("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        return None
    
def save_model(model:lgb.LGBMClassifier,path:str) -> None:
    """Save the trained model to specified path."""
    try:
        with open(path,'wb') as f:
            pickle.dump(model,f)
        logger.debug(f"Model saved successfully at {path}.")
    except Exception as e:
        logger.error(f"Error in saving model: {e}")
        raise

def main():
    try:
        logger.debug("Model building started.")
        root=get_root_directory()
        # load params
        params = load_params("./params.yaml")
        model_params = params.get("model_building", {})
        ngram_range = tuple(model_params.get("ngram_range", [1, 3]))
        max_features = model_params.get("max_features", 1000)
        learning_rate = model_params.get("learning_rate", 0.09)
        max_depth = model_params.get("max_depth", 20)
        n_estimators = model_params.get("n_estimators", 367)

        # load the preprocessed training data from the interim directory
        train_data = load_data_interim(os.path.join(root,'data/interim/train_processed.csv'))
        X_train, y_train = apply_tfidf(train_data, ngram_range, max_features)
        if X_train is None or y_train is None:
            logger.error("TF-IDF failed, exiting.")
            return
        model = train_model(X_train, y_train, learning_rate, max_depth, n_estimators)
        if model is None:
            logger.error("Model training failed, exiting.")
            return
        save_model(model, os.path.join(root,'lgbm_model.pkl'))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()