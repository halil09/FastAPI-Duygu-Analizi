import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

def preProcess_data(text): #cleaning the data
    
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

app = FastAPI()

data = pd.read_csv('hepsiburada.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['Review'].values)



def my_pipeline(text):#pipeline
  text_new = preProcess_data(text)
  X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
  X = pad_sequences(X, maxlen=59)
  return X


@app.get('/') #basic get view
def basic_view():
    return {"WELCOME": "GO TO /docs route, or /post or send post request to /predict "}



@app.get('/predict', response_class=HTMLResponse) #data input by forms
def take_inp():
    return '''<form method="post"> 
    <input type="text" maxlength="59" name="text" value="Duygu Dedektörü"/>  
    <input type="submit"/> 
    </form>'''



@app.post('/predict') #prediction on data
def predict(text:str = Form(...)): #input is from forms
    clean_text = my_pipeline(text) #cleaning and preprocessing of the texts
    loaded_model = tf.keras.models.load_model('duygu.h5') #loading the saved model
    predictions = loaded_model.predict(clean_text) #making predictions
    sentiment = int(np.argmax(predictions)) #index of maximum prediction
    probability = max(predictions.tolist()[0]) #probability of maximum prediction
    if probability<0.5: #assigning appropriate name to prediction
        t_sentiment = 'Olumsuz'
    elif probability>0.5:
        t_sentiment = 'Olumlu'
    
    return { #returning a dictionary as endpoint
        "Girdiğiniz cümle": text,
        "Duygu Durumu": t_sentiment,
        "Duygu Yüzdesi": probability
    }