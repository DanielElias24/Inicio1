import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.models import load_model
import keras
import h5py
import requests
import os
from streamlit.components.v1 import html

#print(os.listdir("../input"))

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("de_chat.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.colorhexa.com/191b20.png");
background-size: 100%;
background-position: top right;
background-repeat: repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
background-size: 150%;
background-position: top left; 
background-repeat: repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def show_page():

 data_price =  pd.read_csv("prices.csv")
 data_fundament = pd.read_csv("fundamentals.csv")
 data_security = pd.read_csv("securities.csv")
 data_price_split = pd.read_csv("prices-split-adjusted.csv", index_col='date', parse_dates=['date'])

 st.title("Tendencias Financieras")

 st.write(":green[*Modelo ML - RNN - LSTM*]")

 st.subheader("Exploración y Análisis")

 st.write("Los datos presentados provienen de la Bolsa de Nueva York, por lo que contienen información financiera, es un set de datos en los cuales vienen 'Fundamentos' que contiene información como ingresos, ganancias, activos, deudas, entre otros, también esta 'prices-split-adjusted' que contiene los precios ajustados por split o dividendo de las acciones, que contiene la información para que se refleje el rendimiento real de las acciones, 'precios' contiene lo precios de las acciones sin splits, 'seguros' contiene informaciones variadas como información de la empresa, símbolo bursátil, el sector la industria, etc.")

 st.subheader("Manipulación y Limpieza")

 st.write("Se pueden mostrar las datas explícitamente y si tienen datos nulos:")

 st.write("-Se muestra la tabla price.csv y sus datos nulos")

 st.write(data_price.head(5))


 st.write(pd.DataFrame(data_price.isna().sum()).T)
 st.write(data_price.shape)

 st.write("-Se muestra la tabla fundamentals.csv y sus datos nulos")

 st.write(data_fundament.head(5))


 st.write(pd.DataFrame(data_fundament.isna().sum()).T)
 st.write(data_fundament.shape)

 st.write("-Se muestra la tabla securities.csv y sus datos nulos")

 st.write(data_security.head(5))


 st.write(pd.DataFrame(data_security.isna().sum()).T)
 st.write(data_security.shape)

 st.write("-Se muestra la tabla prices-split-adjusted.csv y sus datos nulos")

 st.write(data_price_split.head(5))


 st.write(pd.DataFrame(data_price_split.isna().sum()).T)
 st.write(data_price_split.shape)

 st.write("Son pocos los datos nulos, en general esas columnas no se ocupan, así que no hay necesidad de limpiar.")

 st.subheader("Ingeniería de caracterísitcas")


 st.write("En este proyecto no se ocuparan las cuatro tablas sino que se ocupara solo 'prices-split-adjusted.csv', pero esta data se puede explicar con las demás tablas. En específico veremos cuales son los valores en la bolsa con los ajustes a los split de 'APPL'.")

 st.write(data_security[data_security["Ticker symbol"]=="AAPL"])

 st.write("El símbolo bursátil 'AAPL' corresponde a Apple Inc. se analizarán solo los datos de esta empresa tecnológica  y sus valores ajustados con splits.")

 st.code("""
   data_aapl = data_price_split[data_price_split.symbol == 'AAPL']
   data_aapl.drop(['symbol'],1,inplace=True)
   data_aapl["date"] = data_aapl.index  
   data_aapl['date'] = pd.to_datetime(data_aapl['date'])
   print(data_aapl.head())
   print(data_aapl.shape)
   """)

 data_aapl = data_price_split[data_price_split.symbol == 'AAPL']
 data_aapl.drop(['symbol'],1,inplace=True)
#data_aapl["date"] = data_aapl.index  
#data_aapl['date'] = pd.to_datetime(data_aapl['date'])
 st.write(data_aapl.head())
 st.write(data_aapl.shape)

 st.write("Visualizamos como se ven las funciones de 'close'.")

 st.code("""
    fig, ax=plt.subplots()
    sns.set(style="darkgrid")
    sns.lineplot(x=data_aapl["date"], y=data_aapl["close"])
    plt.ylabel("Precio de cierre")
    plt.xlabel("Dias")
    plt.title("Valor del precio de cierre por día")
    plt.show()
   """)

#st.write(data_aapl.shape)

 fig, ax=plt.subplots()
 sns.set(style="darkgrid")
 sns.lineplot(x=data_aapl.index, y=data_aapl["close"])
 plt.ylabel("Precio de cierre")
 plt.xlabel("Dias")
 plt.title("Precio de cierre por día de Apple Inc.")
 st.pyplot(fig)

 st.write("Podemos ver como se comparan los cuatro valores en el tiempo.")

 st.code("""
   fig, ax=plt.subplots()
   sns.set(style="darkgrid")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["close"], label="cierre")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["open"], label="apertura")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["low"], label="bajo")
   sns.lineplot(x=data_aapl["date"], y=data_aapl["high"], label="alto")
   plt.ylabel("Precio de cierre")
   plt.xlabel("Dias")
   plt.title("Precio de cierre por día")
   plt.show()
 """)

 fig, ax=plt.subplots()
 sns.set(style="darkgrid")
 sns.lineplot(x=data_aapl.index, y=data_aapl["close"], label="de cierre")
 sns.lineplot(x=data_aapl.index, y=data_aapl["open"], label="de apertura")
 sns.lineplot(x=data_aapl.index, y=data_aapl["low"], label="más bajo")
 sns.lineplot(x=data_aapl.index, y=data_aapl["high"], label="más alto")
 plt.ylabel("Precios")
 plt.xlabel("Años")
 plt.title("Precios diarios de Apple Inc.")
 st.pyplot(fig)

 st.write("Se ve que tanto los valores de cierre, apertura, bajo y alto tienen valores bastantes similares, esto es muy util, ya que las predicciones pueden ser mejores.")

 st.subheader("-Creación y ajuste de red neuronal")

 st.write("Una red neuronal dedicada para pronóstico de continuación de funciones, es el caso de RNN con el modelo de memoria a largo plazo como LSTM.")

 st.write("El modelo LSTM es muy sensible a la escalabilidad de los datos, por lo que deben este todos en los escalados, también se le agrega una columna nueva con shift 1 para la secuencia de entrenamiento.")

 st.code("""
  from sklearn.preprocessing import MinMaxScaler
  data_df=data_aapl
  columna_objetivo = 'close'
  data_aapl['close_shift'] = data_aapl[columna_objetivo].shift(1)
  data_aapl.dropna(inplace=True) 
  print(data_aapl.head())
  min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
  dataset = min_max_scaler.fit_transform(data_aapl['close'].values.reshape(-1, 1))
  """)
  
 data_df=data_aapl
 columna_objetivo = 'close'
 data_df['close_shift'] = data_df[columna_objetivo].shift(1)
 data_df.dropna(inplace=True) 
 print(data_df.head())

 min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
 dataset = min_max_scaler.fit_transform(data_df['close'].values.reshape(-1, 1))

 st.write("Además se debe separar manualmente los datos de entrenamiento y de prueba para conservar el orden especifico.")

 st.code("""
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
 """)

 train_size = int(len(dataset) * 0.7)
 test_size = len(dataset) - train_size
 train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
 st.write(len(train), len(test))

 st.write("Se debe crear una secuencia para practicar un modelo LSTM, esta secuencia debe estar movida para funcionar, creará un nuevo array que contenga cantidad de secuencias que le especifiquemos. en este caso es 15, los datos los ira agrupando en listas, los datos 'datosX' serán para separarlos en datos de práctica y datosY para separarlos en datos de prueba.")

 st.code("""
    def create_dataset(dataset, secuencia=15):
    datosX, datosY = [], []
    for i in range(len(dataset)-secuencia-1):
        a = dataset[i:(i+secuencia), 0]
        datosX.append(a)
        datosY.append(dataset[i + secuencia, 0])
    return np.array(datosX), np.array(datosY)
    x_train, y_train = create_dataset(train, secuencia=15)
    x_test, y_test = create_dataset(test, secuencia=15)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])) 
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
 """)

 def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

 x_train, y_train = create_dataset(train, look_back=15)
 x_test, y_test = create_dataset(test, look_back=15)

 x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
 x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))



 look_back = 15

 st.subheader("-Modelo LSTM anidado")

 st.write("Se crear un modelo LSTM anidado, en este caso con una sola columna de salida, con error cuadrático medio y optimizador adam, con 20 neuronas por capa.")

 st.code("""
   model = Sequential()
   model.add(LSTM(20, return_sequences=True, input_shape=(1, 15)))
   model.add(LSTM(20))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error', optimizer='adam')
   model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
   trainPredict = model.predict(x_train)
   testPredict = model.predict(x_test)
   trainPredict = min_max_scaler.inverse_transform(trainPredict)
   trainY = min_max_scaler.inverse_transform([y_train])
   testPredict = min_max_scaler.inverse_transform(testPredict)
   testY = min_max_scaler.inverse_transform([y_test])
 """)

 model = Sequential()
 model.add(LSTM(20, return_sequences=True, input_shape=(1, 15)))
 model.add(LSTM(20))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')

 st.markdown("""
 <style>.stSpinner > div > div {
     border-top-color: #0f0;
 }</style>
 """, unsafe_allow_html=True)
 with st.spinner("Entrenando el modelo, espere un momento..."):
   model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)




 trainPredict = model.predict(x_train)
 testPredict = model.predict(x_test)




 trainPredict = min_max_scaler.inverse_transform(trainPredict)
 trainY = min_max_scaler.inverse_transform([y_train])
 testPredict = min_max_scaler.inverse_transform(testPredict)
 testY = min_max_scaler.inverse_transform([y_test])



 trainPredictPlot = np.empty_like(dataset)
 trainPredictPlot[:, :] = np.nan
 trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

 testPredictPlot = np.empty_like(dataset)
 testPredictPlot[:, :] = np.nan
 testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


 fig, ax=plt.subplots()
 sns.set(style="darkgrid")

 plt.plot(trainPredictPlot)
 plt.plot(testPredictPlot)
 plt.ylabel("Precios")
 plt.xlabel("Días")
 plt.title("Precios diarios pronosticados de Apple Inc.")
 st.pyplot(fig)


 st.code("""
   trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
   testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
 """)

 trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

 testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

 st.write('Train Score: %.2f RMSE' % (trainScore))
 st.write('Test Score: %.2f RMSE' % (testScore))


###################################################Parte 2

 st.subheader("-Rango de predicciones")

 st.write("Los datos entrenamientos son únicos por lo que siguen solo una secuencia, esto hace que se considere una posibilidad de predicciones de manera que puede ser inexacta, una forma correcta de tomar esta predicción es tomar un rango, este rango se puede tomar de las otras mediciones diarias del mismo rendimiento, esto hace que las predicciones consideren más posibilidades, esto de mayor seguridad al tomar decisiones, ya que considera el valor más alto y el valor más bajo.")

 data_df=data_aapl
 columna_objetivo = 'low'
 data_df['low_shift'] = data_df[columna_objetivo].shift(1)
 data_df.dropna(inplace=True) 
 print(data_df.head())

 min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
 dataset = min_max_scaler.fit_transform(data_df['low'].values.reshape(-1, 1))


 train_size = int(len(dataset) * 0.7)
 test_size = len(dataset) - train_size
 train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


 def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

 x_train, y_train = create_dataset(train, look_back=15)
 x_test, y_test = create_dataset(test, look_back=15)

 x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
 x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))



 look_back = 15


 model = Sequential()
 model.add(LSTM(20, return_sequences=True, input_shape=(1, 15)))
 model.add(LSTM(20))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')

 st.markdown("""
 <style>.stSpinner > div > div {
    border-top-color: #0f0;
 }</style>
 """, unsafe_allow_html=True)
 with st.spinner("Obteniendo predicciones de valores más bajos, espere un momento..."):
   model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


 trainPredict2 = model.predict(x_train)
 testPredict2 = model.predict(x_test)




 trainPredict2 = min_max_scaler.inverse_transform(trainPredict2)
 trainY2 = min_max_scaler.inverse_transform([y_train])
 testPredict2 = min_max_scaler.inverse_transform(testPredict2)
 testY2 = min_max_scaler.inverse_transform([y_test])



 trainPredictPlot2 = np.empty_like(dataset)
 trainPredictPlot2[:, :] = np.nan
 trainPredictPlot2[look_back:len(trainPredict2)+look_back, :] = trainPredict2

 testPredictPlot2 = np.empty_like(dataset)
 testPredictPlot2[:, :] = np.nan
 testPredictPlot2[len(trainPredict2)+(look_back*2)+1:len(dataset)-1, :] = testPredict2



 data_df=data_aapl
 columna_objetivo = 'high'
 data_df['high_shift'] = data_df[columna_objetivo].shift(1)
 data_df.dropna(inplace=True) 

 min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
 dataset = min_max_scaler.fit_transform(data_df['high'].values.reshape(-1, 1))


 train_size = int(len(dataset) * 0.7)
 test_size = len(dataset) - train_size
 train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


 def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

 x_train, y_train = create_dataset(train, look_back=15)
 x_test, y_test = create_dataset(test, look_back=15)

 x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
 x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))



 look_back = 15


 model = Sequential()
 model.add(LSTM(20, return_sequences=True, input_shape=(1, 15)))
 model.add(LSTM(20))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')

 st.markdown("""
 <style>.stSpinner > div > div {
    border-top-color: #0f0;
 }</style>
 """, unsafe_allow_html=True)
 with st.spinner("Obteniendo predicciones de valores más altos, espere un momento..."):
   model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


 trainPredict3 = model.predict(x_train)
 testPredict3 = model.predict(x_test)




 trainPredict3 = min_max_scaler.inverse_transform(trainPredict3)
 trainY3 = min_max_scaler.inverse_transform([y_train])
 testPredict3 = min_max_scaler.inverse_transform(testPredict3)
 testY3 = min_max_scaler.inverse_transform([y_test])



 trainPredictPlot3 = np.empty_like(dataset)
 trainPredictPlot3[:, :] = np.nan
 trainPredictPlot3[look_back:len(trainPredict3)+look_back, :] = trainPredict3

 testPredictPlot3 = np.empty_like(dataset)
 testPredictPlot3[:, :] = np.nan
 testPredictPlot3[len(trainPredict3)+(look_back*2)+1:len(dataset)-1, :] = testPredict3



 data_df=data_aapl
 columna_objetivo = 'open'
 data_df['open_shift'] = data_df[columna_objetivo].shift(1)
 data_df.dropna(inplace=True) 

 min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
 dataset = min_max_scaler.fit_transform(data_df['open'].values.reshape(-1, 1))


 train_size = int(len(dataset) * 0.7)
 test_size = len(dataset) - train_size
 train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


 def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

 x_train, y_train = create_dataset(train, look_back=15)
 x_test, y_test = create_dataset(test, look_back=15) 

 x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
 x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

#x_train2 = x_train[:-15, :, :]

 look_back = 15


 model = Sequential()
 model.add(LSTM(20, return_sequences=True, input_shape=(1, 15)))
 model.add(LSTM(20))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam')

 st.markdown("""
 <style>.stSpinner > div > div {
     border-top-color: #0f0;
 }</style>
 """, unsafe_allow_html=True)
 with st.spinner("Obteniendo predicciones de apertura, espere un momento..."):
   model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


 trainPredict4 = model.predict(x_train)
 testPredict4 = model.predict(x_test)



# invert predictions
 trainPredict4 = min_max_scaler.inverse_transform(trainPredict4)
 trainY4 = min_max_scaler.inverse_transform([y_train])
 testPredict4 = min_max_scaler.inverse_transform(testPredict4)
 testY4 = min_max_scaler.inverse_transform([y_test])



 trainPredictPlot4 = np.empty_like(dataset)
 trainPredictPlot4[:, :] = np.nan
 trainPredictPlot4[look_back:len(trainPredict4)+look_back, :] = trainPredict4

 testPredictPlot4 = np.empty_like(dataset)
 testPredictPlot4[:, :] = np.nan
 testPredictPlot4[len(trainPredict4)+(look_back*2)+1:len(dataset)-1, :] = testPredict4




 fig, ax=plt.subplots()
 sns.set(style="darkgrid")
 plt.plot(trainPredictPlot)
 plt.plot(testPredictPlot, label="Predicción de cierre")
 plt.plot(testPredictPlot2, label="Predicción de valor bajo")
 plt.plot(testPredictPlot3, label="Predicción de valor alto")
 plt.plot(testPredictPlot4, label="Predicción de apertura")
 plt.ylabel("Precios")
 plt.xlabel("Días")
 plt.legend()
 plt.title("Rango de predicciones de Apple Inc.")
 st.pyplot(fig)

 st.subheader("Conclusiones")

 st.write("El modelo LSTM anidado hace una buena representación de los datos en sus predicciones, en general tiene una buena interpretación de tendencias, tiene mucha acertividad en ese aspecto, en resultado bastante cercano a los valores reales. Tiene algunas impresiciones en el tamaño de los picos o en los valles, pero en ese sentido el rango daria mas seguridad a la hora de tomar decisiones.")




