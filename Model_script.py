import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM,Dense
import nltk
import re
from nltk.corpus import stopwords
from sklearn import preprocessing
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
df=pd.read_csv("/Users/santh/Downloads/project3/results.csv",sep="|")
new_df=df[0:15000]
label_encoder = preprocessing.LabelEncoder()
  

new_df['image_no']= label_encoder.fit_transform(new_df['image_name'])
x=new_df[' comment']
y=new_df['image_no']
message=x.copy()
newwords=[]
ps= PorterStemmer()
for i in range(0, len(message)):
  review= re.sub('[^a-zA-Z]',' ',message[i])
  review= review.lower()
  review= review.split()
  checking=[ps.stem(word) for word in review if word not in stopwords.words('english')]
  checking=' '.join(checking)
  newwords.append(checking)
onehot_rep=[one_hot(word,5000) for word in newwords]
emb_doc=pad_sequences(onehot_rep,padding='pre',maxlen=20)
model=Sequential()
model.add(Embedding(5000,40,input_length=20))
model.add(LSTM(100))
model.add(Dense(3000,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
x=np.array(emb_doc)
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=2)
model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs=50 ,batch_size=12)
model.save('NLP_model_final_2.h5')


