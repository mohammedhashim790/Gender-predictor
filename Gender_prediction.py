#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelBinarizer


# In[2]:


#read csv using open window for windows OS
root = tk.Tk()
root.withdraw()
path = askopenfilename()
data = pd.read_csv(path)
le = LabelBinarizer()


# In[3]:


print("[INFO].....Sample files from dataset : \n{}".format(data.head()))
X = data.Name
Y = data.Gender
#X = data.name
#Y = data.label


# In[4]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 53)


# In[5]:


cnt_vec = CountVectorizer(stop_words='english')
cnt_train = cnt_vec.fit_transform(x_train)
cnt_test = cnt_vec.transform(x_test)
#print(cnt_test)
#print(Y)


# In[6]:


#print(cnt_vec.get_feature_names()[:10])


# In[7]:


nb_cls = MultinomialNB()
nb_cls.fit(cnt_train,y_train)


# In[8]:


#to print accuracy score
pred = nb_cls.predict(cnt_test)
print("Model's accuracy score : {}".format(accuracy_score(y_test,pred)))


# In[11]:


#sample Testing
s = input("Enter your name : ")
test = cnt_vec.transform([s])
n_p = nb_cls.predict(test)
print("The model thinks the your gender is : {}".format(n_p[0]))


# In[ ]:




