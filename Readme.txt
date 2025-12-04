In this project, we did experiments using MLFlow to explore the better ML model on Jupyter Workbook. EC2 and S3 are needed on AWS.
After that, we build the dvc pipeline and Python codes, to automatically process data, build model, train model and register model
We built a Flask Api and a frontend. (Google Cloud Api key needed) Then we upload the unpacked folder to Chrome extensions to add the plug-in.
Finally, we look at Docker and K8s for CICD.


To be able to access MLFlow, we need to set up the server (virtual machine and volume) on ACW

Then on the virtual machine, after installing all the dependencies, we need to run pipenv: run pipenv shell

After that, supposing that we have configured the inbound rules for TCP in the group securities on ACW, to expose the MLFlow page, we can run: 
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root s3://mlflow-bucket-fernando --allowed-hosts "*"
mlflow server --backend-store-uri sqlite:///~/mlflow.db --default-artifact-root s3://mlflow-bucket-fernando --host 0.0.0.0 --port 5000 --allowed-hosts "*"


conda create -n youtube python=3.11 -youtube
conda activate youtube
python -m venv youtube
.\youtube\Scripts\Activate.ps1


pip install -r requirements.txt
dvc init
dvc repro
dvc dag
aws configure


AWS
aws configure

Json data demo in postman
http://localhost:5000/predict

{
    "comments": ["This video is awsome! I loved a lot", "Very bad explanation. poor video"]
}
chrome://extensions

how to get youtube api key from gcp:
https://www.youtube.com/watch?v=i_FdiQMwKiw

AWS-CICD-Deployment-with-Github-Actions
1. Login to AWS console.
2. Create IAM user for deployment
#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess
3. Create ECR repo to store/save docker image
- Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/youtube
4. Create EC2 machine (Ubuntu)
5. Open EC2 and Install docker in EC2 Machine:
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
6. Configure EC2 as self-hosted runner:
setting>actions>runner>new self hosted runner> choose os> then run command one by one
7. Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = simple-app