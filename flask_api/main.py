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

app=Flask(__name__)
CORS(app) # Enable CORS for all routes

model=load_model('lgbm_model.pkl')
# model=load_model_from_registry('yt_chrome_plugin_model',"1")
vectorizer=load_vectorizer('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "Sentiment Insights API is running."

def process_comments_return_predictions_and_sentiments(comments):
    print("Preprocessing comments...")
    clean_comments=[preprocess_comment(comment) for comment in comments]
    print("Transforming comments...")
    X=vectorizer.transform(clean_comments) # sparse matrix
    feature_names = vectorizer.get_feature_names_out()
    # convert to dense array
    dense=X.toarray()   
    dense_df=pd.DataFrame(dense, columns=feature_names)
    print("TF-IDF transformation complete.")
    # Predict using the loaded model
    predictions=model.predict(dense_df).tolist()
    sentiment_map={-1:'Negative',0:'Neutral',1:'Positive'}
    sentiments=[sentiment_map.get(p, 'Unknown') for p in predictions]
    return predictions, sentiments

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.json
        comments=data.get('comments','')
        if isinstance(comments, str):
            comments = [comments]

        print("this is the comment",comments,type(comments))
        if not comments:
            return jsonify({'error':'No comment provided'}),400
        
        predictions, sentiments =  process_comments_return_predictions_and_sentiments(comments)
    except Exception as e:
        return jsonify({'error':str(e)}),500
    response=[{'comment':comments[i],'prediction':predictions[i],'sentiment':sentiments[i]} for i in range(len(comments))]
    return jsonify(response)

@app.route('/predict_with_timestamps',methods=['POST'])
def predict_with_timestamps():
    try:
        data=request.json
        comments_data=data.get('comments','')
        if not comments_data:
            return jsonify({'error':'No comment provided'}),400
        comments=[d['text'] for d in comments_data]
        timestamps=[d['timestamp'] for d in comments_data]
        predictions, sentiments =  process_comments_return_predictions_and_sentiments(comments)
    except Exception as e:
        return jsonify({'error':str(e)}),500
    response=[{'comment':comments[i],'prediction':str(predictions[i]),'sentiment':str(predictions[i]),'timestamp':timestamps[i]} for i in range(len(comments))]
    print(response[0:2])
    return jsonify(response)

@app.route('/generate_chart',methods=['POST'])
def generate_chart():
    try:
        data=request.get_json()
        sentiment_counts=data.get('sentiment_counts',{})
        if not sentiment_counts:
            print("No sentiment counts provided",data)
            return jsonify({'error':'No sentiment counts provided'}),400
        # prepare data fro the pie chart
        labels=['Positive','Neutral','Negative']
        sizes=[
            int(sentiment_counts.get('1',0)),
            int(sentiment_counts.get('0',0)),
            int(sentiment_counts.get('-1',0))
        ]
        if sum(sizes)==0:
            print("Sentiment counts are all zero",data,sizes)
            return jsonify({'error':'Sentiment counts are all zero'}),400
        colors=['#4CAF50','#FFC107','#F44336'] # green, amber, red
        plt.figure(figsize=(6,6))
        plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',startangle=140,textprops={'fontsize': 14,'color':'white'})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # Save the plot to a BytesIO object
        img_bytes=io.BytesIO()
        plt.savefig(img_bytes,format='png',bbox_inches='tight',facecolor='#2E2E2E',transparent=True)
        img_bytes.seek(0)
        plt.close()
        return send_file(img_bytes, mimetype='image/png')
    except Exception as e:
        return jsonify({'error':str(e)}),500
    

@app.route('/generate_wordcloud',methods=['POST'])
def generate_wordcloud():
    try:
        data=request.get_json()
        comments=data.get('comments','')
        if not comments:
            return jsonify({'error':'No comments provided'}),400
        text=" ".join(comments)
        wordcloud=WordCloud(width=800,height=400,background_color='black',colormap='viridis',stopwords=set(stopwords.words('english')),collocations=False).generate(text)
        img_bytes=io.BytesIO()
        wordcloud.to_image().save(img_bytes,format='PNG')
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype='image/png')
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/generate_trend_graph',methods=['POST'])
def generate_trend_graph():
    try:
        data=request.get_json()
        sentiment_data=data.get('sentiment_data','')
        if not sentiment_data:
            return jsonify({'error':'No sentiment data provided'}),400
        df=pd.DataFrame(sentiment_data)
        df['timestamp']=pd.to_datetime(df['timestamp'])
        df.set_index('timestamp',inplace=True)
        df['sentiment'] = df['sentiment'].astype(int) # ensure prediction is int
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        # Resample the data over monthly intervals and count sentiments - since the index is set on the timestamp now, we can regroup by month
        monthly_counts=df.resample('M')['sentiment'].value_counts().unstack(fill_value=0) # months as rows, sentiment classes as columns
        monthly_total=monthly_counts.sum(axis=1) # axis=0 is rows, axis=1 is columns, sum across columns to get total per month
        monthly_pcs=(monthly_counts.T / monthly_total).T * 100 # transpose, divide, transpose back, the division tries to match the number of columns (not number of rows) to broacast

        # ensure all sentiments are present in the columns
        for sentiment in sentiment_labels.values():     
            if sentiment not in monthly_counts.columns:
                monthly_counts[sentiment]=0
        plt.figure(figsize=(10,6))
        for pred_v in [-1,0,1]:
            if pred_v not in monthly_pcs.columns:
                monthly_pcs[pred_v]=0
        # sort columns by sentiment value
        monthly_pcs=monthly_pcs[[-1,0,1]]

        plt.figure(figsize=(12,6))
        colors={-1:'#F44336',0:'#FFC107',1:'#4CAF50'} # red, amber, green
        for pred_v in [-1,0,1]:
            plt.plot(monthly_pcs.index,monthly_pcs[pred_v],marker='o',linestyle='-',color=colors[pred_v],label=sentiment_labels[pred_v])
        plt.xlabel('Month',fontsize=14)
        plt.ylabel('Pc of Comments (%)',fontsize=14)
        plt.title('Sentiment Trend Over Time',fontsize=16)
        
        plt.grid(True)
        plt.xticks(rotation=45)
        # format the x-axis dates
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(dates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        img_bytes=io.BytesIO()
        plt.savefig(img_bytes,format='png',bbox_inches='tight',facecolor='#2E2E2E',transparent=True)
        img_bytes.seek(0)
        plt.close()
        return send_file(img_bytes, mimetype='image/png')
    except Exception as e:
        return jsonify({'error':str(e)}),500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)