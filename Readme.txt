To be able to access MLFlow, we need to set up the server (virtual machine and volume) on ACW

Then on the virtual machine, after installing all the dependencies, we need to run pipenv: run pipenv shell

After that, supposing that we have configured the inbound rules for TCP in the group securities on ACW, to expose the MLFlow page, we can run: mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://mlflow-bucket-fernando --allowed-hosts "*"


conda create -n youtube python=3.11 -youtube
conda activate youtube
python -m venv youtube
.\youtube\Scripts\activate.bat


pip install -r requirements.txt
dvc init
dvc repro
dvc dag
aws configure