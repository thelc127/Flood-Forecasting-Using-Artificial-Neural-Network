# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:38:25 2019

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:17:04 2019

@author: asus
"""

from tkinter import *
import tkinter.messagebox

# Empty list to append IDS
ids =[]

# tkniter window

class Application:
    def __init__(self,master):
        
        import tensorflow as tf
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        #importing the datasets
        dataset = pd.read_csv('manaslu.csv')
        X = dataset.iloc[:,1:6].values # inputs
        y = dataset.iloc[:,7].values # outputs
        
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
        
        #featurescaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        ################## finished data preprocessing #############
        
        
        #### Evaluating the  ANN
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV
        
        ############# Making the ANN #################
        from keras.models import Sequential # initialize our NN
        from keras.layers import Dense # to make different layers
        from keras.layers import Dropout # applied so that some neurons can be randomly be disabled helps to reduce overfitting
    #import keras and packages
    # initializing the ANN
        self.classifier =  Sequential()
    # adding the input layer and first hidden layer
    # dense function will take care of initializing the weight to 0
        self.classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_shape=(5,)))
        # classifier.add(Dropout(rate = 0.1))
    # add second hidden layer
        self.classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
        # classifier.add(Dropout(rate = 0.1))
    # adding the final layer ie output layer
        self.classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    # Compiling the ANN
        self.classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        
        self.classifier.fit(X_train,y_train,batch_size=50,epochs=60)
        
        
        # creating frames in master
        self.left = Frame(master,width = 1280,height = 720,bg = 'skyblue')
        self.left.pack(side=LEFT)

        # labelsfor the window
        self.heading = Label(self.left,text="Flood Prediction System", font=('arial 40 bold'),fg='black',bg='lightgreen')
        self.heading.place(x=100,y=0)

        
     
        # Rainfall in may
        self.rfmay = Label(self.left,text="Cloudiness",font=('arial 18 bold'),fg='black',bg='lightgreen')
        self.rfmay.place(x=0,y=140)
 
        # Rainfall in June
        self.rfjune = Label(self.left,text="Rainfall (mm)",font=('arial 18 bold'),fg='black',bg='lightgreen')
        self.rfjune.place(x=0,y=180)

        # Rainfall in July
        self.rfjuly = Label(self.left,text="Max temperature(°C)",font=('arial 18 bold'),fg='black',bg='lightgreen')
        self.rfjuly.place(x=0,y=220)
        # Average Temperature
        self.temp = Label(self.left,text="Min temperature(°C)",font=('arial 18 bold'),fg='black',bg='lightgreen')
        self.temp.place(x=0,y=260)
        # altitude
        self.alti = Label(self.left,text="Humidity(mm)",font=('arial 18 bold'),fg='black',bg='lightgreen')
        self.alti.place(x=0,y=300)
        
   

        # Entries for all labels

       

        self.rfmay_ent = Entry(self.left,width=30)
        self.rfmay_ent.place(x=350,y=140)

        self.rfjune_ent = Entry(self.left,width=30)
        self.rfjune_ent.place(x=350,y=180)

        self.rfjuly_ent = Entry(self.left,width=30)
        self.rfjuly_ent.place(x=350,y=220)

        self.temp_ent = Entry(self.left,width=30)
        self.temp_ent.place(x=350,y=260)

        self.alti_ent = Entry(self.left,width=30)
        self.alti_ent.place(x=350,y=300)
        
     
        
        # button to perform a command
        self.submit = Button(self.left, text="Submit",width=20,height =2, bg = 'steelblue',command=self.my_predict)
        self.submit.place(x=300,y=380)

############################################################################################################################

############################################################################################################################
   


    def my_predict(self):
        import numpy as np
    
    
        self.val2=self.rfmay_ent.get()
        self.val3=self.rfjune_ent.get()
        self.val4=self.rfjuly_ent.get()
        self.val5=self.temp_ent.get()
        self.val6=self.alti_ent.get()
        abc= np.array([[self.val2,self.val3,self.val4,self.val5,self.val6]])
       # self = Tk()
        xyz = self.classifier.predict(abc)
        self.final = (xyz[0][0]*100)
       
        if  self.val2=='' or self.val3=='' or self.val4=='' or self.val5=='' or self.val6==''  :
            tkinter.messagebox.showwarning("Warning","Please fill up All boxes")
        else:
           # Entry(self,  text = "%.2f" %(self.final) ).grid(row=2, column=1)
            tkinter.messagebox.showinfo("Success","The probability of flood is  "  +str(self.final))

# creating the object
root = Tk()
b = Application(root)

# resolution of window
root.geometry("1280x720+0+0")

# preventing the resize feature
root.resizable(False,False)

root.mainloop()
