import numpy as np
import pandas as pd
import os,re,nltk,string,logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# logging configuration
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')
file_handler=logging.FileHandler('data_preprocessing.log')
file_handler.setLevel('ERROR')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a single comment."""
    try:
        # Convert to lowercase and remove trailling and leading whitespaces
        comment = comment.lower().strip()
        
        # Clean up the string
        comment = re.sub(r'\n',' ',comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]+', '', comment)
        
        # Remove stopwords + lemmatization
        stop_words = set(stopwords.words('english'))-{'not','no''but','however','yet','against'}
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in comment.split() if word not in stop_words]

        # Join tokens back to string
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return ""
    
def apply_preprocessing_to_df(df):
    try:
        df['clean_comment']=df['clean_comment'].apply(preprocess_comment)
        logger.debug("Preprocessing applied successfully to DataFrame.")
        return df
    except Exception as e:
        logger.error(f"Error in applying preprocessing to DataFrame: {e}")
        return df
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,path:str) -> None:
    """Save the processed data to specified paths."""
    try:
        interim_path=os.path.join(path,'interim')
        os.makedirs(interim_path,exist_ok=True)
        train_path=os.path.join(interim_path,'train_processed.csv')
        test_path=os.path.join(interim_path,'test.csv')
        train_data.to_csv(train_path,index=False)
        test_data.to_csv(test_path,index=False)
        logger.debug(f"Data saved successfully at {train_path} and {test_path}.")
    except Exception as e:
        logger.error(f"Error in saving data: {e}")

def main():
    try:
        logger.debug("Data preprocessing started.")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        train_processed=apply_preprocessing_to_df(train_data)
        test_processed=apply_preprocessing_to_df(test_data)
        save_data(train_processed,test_processed,'./data')
    except Exception as e:
        logger.error(f"Error in main preprocessing function: {e}")

if __name__=='__main__':
    main()
