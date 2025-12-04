import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for environments without display
from flask import Flask, request,jsonify,send_file
from flask_cors import CORS
import io,mlflow,re,pickle,sys,os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as dates
from wordcloud import WordCloud

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
print("PYTHONPATH añadido:", PROJECT_ROOT)
from src.data.data_preprocessing import preprocess_comment
from src.utils.load import load_model_from_registry,load_vectorizer,load_model
from flask_api.main import home,process_comments_return_predictions_and_sentiments,predict,predict_with_timestamps,generate_chart,generate_wordcloud,generate_trend_graph

app=Flask(__name__)
CORS(app) # Enable CORS for all routes

model=load_model('lgbm_model.pkl')
# model=load_model_from_registry('yt_chrome_plugin_model',"1")
vectorizer=load_vectorizer('tfidf_vectorizer.pkl')

@app.route('/')
def home_app():
    return home()

@app.route('/predict',methods=['POST'])
def predict_app():
   return predict() 

@app.route('/predict_with_timestamps',methods=['POST'])
def predict_with_timestamps_app():
    return predict_with_timestamps()

@app.route('/generate_chart',methods=['POST'])
def generate_chart_app():
    return generate_chart()
    

@app.route('/generate_wordcloud',methods=['POST'])
def generate_wordcloud_app():
    return generate_wordcloud()

@app.route('/generate_trend_graph',methods=['POST'])
def generate_trend_graph_app():
    return generate_trend_graph()

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)