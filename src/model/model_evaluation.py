import numpy as np
import pandas as pd
import pickle,logging,yaml,mlflow,mlflow.sklearn,os,json
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
from src.utils.logger import get_logger
from src.utils.commonFuncs import get_root_directory
from src.utils.load import load_params,load_data_interim,load_model,load_vectorizer

# logging
logger=get_logger("model_evaluation")

def evaluate_model(model, X_test:np.ndarray, y_test:np.ndarray):
    """Evaluate the model and return classification report and confusion matrix."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed successfully.")
        return report, cm
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        return {}, None
    
def log_confusion_matrix(cm:np.ndarray, dataset_name) -> None:
    """Log confusion matrix as a heatmap."""
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {dataset_name}')
    cm_file_path=f'cm_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()
    logger.debug(f"Confusion matrix saved as {cm_file_path}.")

def save_model_info(run_id:str,model_path:str,file_path:str):
    """Save model information including run ID and model path to a JSON file."""
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }
        with open(file_path, 'w') as f:
            json.dump(model_info, f,indent=4)
        logger.debug(f"Model information saved successfully at {file_path}.")
    except Exception as e:
        logger.error(f"Error in saving model information: {e}")

def main():
    params = load_params("./params.yaml")
    tracking_url = params.get("global", {}).get("tracking_url", "http://ec2-15-222-29-84.ca-central-1.compute.amazonaws.com:5000/")
    experiment_name = params.get("global", {}).get("experiment_name", "dvc-pipeline-runs")
    mlflow.set_tracking_uri(tracking_url)
    mlflow.set_experiment(experiment_name)
    logger.debug("Model evaluation started.")
    root=get_root_directory()
    
        

    with mlflow.start_run() as run:
        try:
            for key, value in params.items():
                if key=="global":
                    continue
                if isinstance(value, dict):
                    for k, v in value.items():
                        mlflow.log_param(f"{key}_{k}", v)
                else:
                    mlflow.log_param(key, value)
            logger.debug("Parameters logged to MLflow.")
            model=load_model(os.path.join(root,'lgbm_model.pkl'))
            vectorizer=load_vectorizer(os.path.join(root,'tfidf_vectorizer.pkl'))
            test_data=load_data_interim(os.path.join(root,'data','interim','test.csv'))
            if test_data.empty:
                logger.error("Test data is empty!")
                return
            y_test=test_data['category'].values
            X_test=vectorizer.transform(test_data['clean_comment'].values)

            # create a df for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test[:5].toarray(), columns=vectorizer.get_feature_names_out())
            # infer signature
            signature = infer_signature(input_example, model.predict(X_test[:5]))
            # log model with signature
            mlflow.sklearn.log_model(model, "model", signature=signature,input_example=input_example)
            # save the model info
            # artifact_uri=mlflow.get_artifact_uri()
            model_path=os.path.join(root, "lgbm_model.pkl")
            mlflow.sklearn.log_model(model, model_path, signature=signature, input_example=input_example)
            save_model_info(run.info.run_id,model_path,os.path.join(root,'model_info.json')) 
            # log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root,'tfidf_vectorizer.pkl'))
            # evaluate the model and get metrics
            report,cm=evaluate_model(model,X_test,y_test)
            for label, metric in report.items():
                if isinstance(metric, dict):
                    for key, value in metric.items():
                        mlflow.log_metric(f"{label}_{key}", value)
            # log cm
            log_confusion_matrix(cm, 'test_dataset')
            # add tags
            mlflow.set_tags({"model_type": "LightGBM", "task":"Sentiment Analysis","dataset": "Youtube Comments"})
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")

if __name__ == "__main__":
    main()