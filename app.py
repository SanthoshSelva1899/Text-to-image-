# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:22:43 2023

@author: santh
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from flask import Flask, redirect, url_for, request, render_template
import tensorflow as tf
import base64
import io
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from sklearn import preprocessing
nltk.download('stopwords')
ps= PorterStemmer()

app = Flask(__name__)

basic='/Users/santh/Downloads/project3/archive (20)/flickr30k_images/flickr30k_images/'
df=pd.read_csv("/Users/santh/Downloads/project3/results.csv",sep="|")
print(df.tail())
new_df=df[0:15000]
label_encoder = preprocessing.LabelEncoder()

MODEL_PATH ='/Users/santh/Downloads/project3/NLP_model_final_2.h5'
model= tf.keras.models.load_model(MODEL_PATH)
  

new_df['image_no']= label_encoder.fit_transform(new_df['image_name'])

@app.route('/', methods=['GET','POST'])
def index():
    # Main page
    return render_template('Index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.form.get('fname')
        print(f)
        newword=[]
        review= re.sub('[^a-zA-Z]',' ',f)
        review= review.lower()
        review= review.split()
        checking=[ps.stem(word) for word in review if word not in stopwords.words('english')]
        checking=' '.join(checking)
        newword.append(checking)
        onehot_rep=[one_hot(word,5000) for word in newword]
        emb_doc=pad_sequences(onehot_rep,padding='pre',maxlen=20)
        x=np.array(emb_doc)
        y=model.predict(x)
        pred=np.argmax(y)
        print(pred)
        output=new_df[new_df['image_no']==pred]['image_name'].unique()
        output2=new_df[new_df['image_no']==pred][' comment']
        print(output)
        print(output2.head())
        for i in output:
          new=str(i)
        path=os.path.join(basic,new)
        img=Image.open(path)
        data=io.BytesIO()
        img.save(data,"JPEG")
        encode_img_data=base64.b64encode(data.getvalue())
        return render_template('Base.html',user_image=encode_img_data.decode("UTF-8"))
if __name__ == '__main__':
    app.run(debug=True)
      