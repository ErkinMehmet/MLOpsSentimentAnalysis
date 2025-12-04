import json,mlflow,os
from src.utils.commonFuncs import get_root_directory
from src.utils.load import load_model_info,load_params
from src.utils.logger import get_logger

# logging configuration
logger = get_logger("model_registration")

def register_model(model_name:str,model_info:dict,client):
    """Register the model in MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)
        logger.debug(f"Model registered successfully with name: {model_name}")
        registered_model=client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

def main():
    try:
        logger.debug("Model registration started.")
        model_info=load_model_info(os.path.join(get_root_directory(),'model_info.json'))
        model_name="yt_chrome_plugin_model"
        # validar que el run exista
        params = load_params("./params.yaml")
        tracking_url = params.get("global", {}).get("tracking_url", "http://ec2-15-222-29-84.ca-central-1.compute.amazonaws.com:5000/")
        experiment_name = params.get("global", {}).get("experiment_name", "dvc-pipeline-runs")
        mlflow.set_tracking_uri(tracking_url)
        mlflow.set_experiment(experiment_name)
        client = mlflow.tracking.MlflowClient()
        try:
            logger.debug(f"Trying to get run_id: {model_info['run_id']}")
            client.get_run(model_info['run_id'])
        except Exception as e:
            # si no existe, tomar el último run del experimento
            logger.debug(f"Failed to get run_id: {model_info['run_id']}")
            experiment = client.get_experiment_by_name("dvc-pipeline-runs")
            runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
            model_info['run_id'] = runs[0].info.run_id
            logger.debug(f"Using last run id: {model_info['run_id']}")
        register_model(model_name,model_info,client)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()