import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import csv
from io import StringIO
print("---------------------------------------------------------------")

st.set_page_config(layout="wide")
dst,iast,rst,pst = st.tabs(["Données", "Paramêtres","Résultats","Prédictions"])
if 'counter' not in st.session_state:
    st.session_state['counter']=0
#------------------------------------------------------dst
k=dst.file_uploader("Upload your data here")


try : 
    byte_content = k.read()
    content = byte_content.decode()
    file = StringIO(content)
    delem=","
    if ";" in str(byte_content):
        delem=";"
    Data=pd.read_csv(file, delimiter=delem)
    dst.dataframe(Data)
    Datanp=Data.to_numpy()
    XX=[]
    YY=[]
    print("TEST")
    col_X=dst.columns([1]*len(Datanp[0]))
    for i in range(len(Datanp[0])):
        XX+=[col_X[i].checkbox("X:"+str(i))]
    col_Y=dst.columns([1]*len(Datanp[0]))
    for i in range(len(Datanp[0])):
        YY+=[col_Y[i].checkbox("Y:"+str(i))]




except :
    m=2

#------------------------------------------------------iast
nb=iast.slider("Couches de neuronnes : ", 0, 20,value=3)
param=iast.columns([1,1,1,1,1],vertical_alignment="center")

try : 
    prevalue="("+str(np.sum(XX))+",1)"
except:
    prevalue="(10,1)"
input_nb=param[0].text_input("Taille Input",value=prevalue)
epoch=param[1].text_input("Epochs",value=30)
loss=param[2].text_input("Loss Function",value="mse")
btc=param[3].number_input("Batch Size",value=1)

iast.text("Le réseau de neuronnes : ")
model = tf.keras.models.Sequential()

def NN():
    global model,X,Y


    # ---------------- Data
    indexx=np.argwhere(np.array(XX)==1)
    indexy=np.argwhere(np.array(YY)==1)
    X=np.transpose([Datanp[:,indexx[0][0]]])
    Y=np.transpose([Datanp[:,indexy[0][0]]])
    for i in range(1,len(indexx)):
        X=np.append(X,np.transpose([Datanp[:,indexx[i][0]]]),axis=1)
    for i in range(1,len(indexy)):
        Y=np.append(Y,np.transpose([Datanp[:,indexy[i][0]]]),axis=1)
    iast.dataframe(pd.DataFrame(X))
    iast.dataframe(pd.DataFrame(Y))
    
    model.fit(np.array(X,dtype=float),np.array(Y,dtype=float),epochs=int(epoch),batch_size=btc)
    iast.text("Modèle Entrainé !")





TDL=[]
col_tab=[]
try:
    prevalueY=np.sum(YY)
except : prevalueY=1
for i in range(nb):
    col_tab+=[iast.columns([1,1,1],vertical_alignment="center")]
    TDL+=[[col_tab[i][0].number_input("Nb Neuronnes",key=str(i)+"number",value=prevalueY),
           col_tab[i][1].text_input("Activation",key=str(i)+"ACT",value="relu"),
           col_tab[i][2].number_input("droprate",key=str(i)+"drop",value=0.)
           ]]
iast.button("TRAIN",on_click=NN)

try:
        # ----------------- MLP
    model.add(tf.keras.layers.Dense(TDL[0][0], kernel_initializer='zeros',input_shape=eval(input_nb), activation=str(TDL[0][1])))
    if TDL[0][2] >0:
        model.add(tf.keras.layers.Dropout(TDL[0][2]))
    for i in range(1,len(TDL)):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(TDL[i][0], activation=TDL[i][1]))
        if TDL[i][2] >0:
            model.add(tf.keras.layers.Dropout(TDL[i][2]))
    model.compile(optimizer='adam',
              loss=loss)
except : 
    m=4
#--------------------- Résultats 


#--------------------- Prédictions 

j=pst.file_uploader("Upload your prediction data here")
try : 
    byte_contentp = j.read()
    contentp = byte_contentp.decode()
    filep = StringIO(contentp)
    delemp=","
    if ";" in str(byte_contentp):
        delemp=";"
    Datap=pd.read_csv(filep, delimiter=delemp)
    pst.dataframe(Datap)
    Datanpp=Datap.to_numpy()
    XXp=[]
    col_Xp=pst.columns([1]*len(Datanpp[0]))
    for i in range(len(Datanpp[0])):
        XXp+=[col_Xp[i].checkbox("Xp:"+str(i))]
    indexxp=np.argwhere(np.array(XXp)==1)
    Xp=np.transpose([Datanpp[:,indexxp[0][0]]])
    for i in range(1,len(indexxp)):
        Xp=np.append(Xp,np.transpose([Datanpp[:,indexxp[i][0]]]),axis=1)
    pst.dataframe(pd.DataFrame(Xp))
except :
    m=2
if model != None:
    if pst.button("Predict : "):

        pst.dataframe(pd.DataFrame(model.predict(np.array(Xp,dtype=float))))




