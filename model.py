import os
import pandas as pd

base_Dir='/Users/santh/Desktop/project4/UTKFace/'

age_list=[]
gender_list=[]

image_path_list=[]

for files in os.list_dir(base_Dir):
    image_path=base_Dir+files
    splitting=files.split("_")
    age= splitting[0]
    gender=splitting[1]
    age_list.append(age)
    gender_list.append(gender)
    image_path_list.append(image_path)
    
    
df=pd.DataFrame()
df['age_list'],df['gender_list'],df['image_path_list']=age_list,gender_list,image_path_list

df.head()