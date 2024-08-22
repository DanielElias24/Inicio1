import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
#import cv2                  
from tqdm import tqdm

import os
from skimage import io, transform                   
from random import shuffle  
from PIL import Image
import base64
import matplotlib.image as mpimg


@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("de_chat.png")

#https://www.blogdelfotografo.com/wp-content/uploads/2021/12/Fondo_Negro_3.webp
#https://w0.peakpx.com/wallpaper/596/336/HD-wallpaper-azul-marino-agua-clima-nubes-oscuro-profundo.jpg
#
#https://i.pinimg.com/564x/8b/5a/a4/8b5aa4783578968cd257b0b5418f3645.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://i.pinimg.com/736x/f6/c1/0a/f6c10a01b8f7c3285fc660b0b0664e52.jpg
#https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjkwNy1hdW0tNDQteC5qcGc.jpg
#https://c.wallhere.com/photos/ac/a0/color_pattern_texture_background_dark-672126.jpg!d

#Colores
#https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg

#https://e0.pxfuel.com/wallpapers/967/154/desktop-wallpaper-solid-black-1920%C3%971080-black-solid-color-background-top.jpg
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

 st.title("Reconocimiento de imágenes")

 st.write(":green[*Modelo ML - CNN*]")

 st.subheader("Exploración y Análisis")

 st.write("En este caso los datos son imágenes en formato '.jpg' que corresponden a flores, estas flores se categorizan en cinco diferentes tipos flores. Las imágenes son en su totalidad de las flores vistas desde diferentes ángulos, iluminaciones y colores, también puede presentar objetos cercanos.")

 st.write("Estos modelos de reconocimiento son útiles cuando se necesitan identificar objetos de forma rápida y de forma automática, cuando son demasiados objetos y un humano se demoraría demasiado o cuando se necesita catalogar un objeto y un humano no está seguro de cual categoría podría ser.")

 st.write("Lo que se busca con este modelo, es que el modelo reconozca que tipo de flores se presentan en la imagen.")

 st.subheader("Manipulación y limpieza")

 st.write("Las imágenes presentan tamaños diferentes por lo que hay que cambiarles el tamaño para que sean exactamente los mismos, en cuanto al tipo de planta, se encuentran ordenadas en diferentes carpetas que las ordenan por tipo.")

 st.write("Lo importante es que tengan el mismo tamaño y saber que tipo son, lo podemos lograr, esto último se puede lograr sabiendo el nombre de la carpeta y la asociamos con el id en otra tupla.")

 st.code("""
    X=[]
    Z=[]
    IMG_SIZE=150
    FLOWER_DAISY_DIR='flowers/daisy'
    FLOWER_SUNFLOWER_DIR='flowers/sunflower'
    FLOWER_TULIP_DIR='flowers/tulip'
    FLOWER_DANDI_DIR='flowers/dandelion'
    FLOWER_ROSE_DIR='flowers/rose'
 """)

 X=[]
 Z=[]
 IMG_SIZE=150
 FLOWER_DAISY_DIR='flowers/daisy'
 FLOWER_SUNFLOWER_DIR='flowers/sunflower'
 FLOWER_TULIP_DIR='flowers/tulip'
 FLOWER_DANDI_DIR='flowers/dandelion'
 FLOWER_ROSE_DIR='flowers/rose'

 st.write("Es necesario definir funciones para que hagan el nuevo tamaño para las imágenes y registre el tipo de la flores en dos tuplas diferentes.")

 st.code("""
    def assign_label(img,flower_type):
       return flower_type

    def make_train_data(flower_type,DIR):
       for img in tqdm(os.listdir(DIR)):
          label=assign_label(img,flower_type)
          path = os.path.join(DIR,img)
          img = cv2.imread(path,cv2.IMREAD_COLOR)
          img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
          X.append(np.array(img))
          Z.append(str(label))
 """)

 def assign_label(img,flower_type):
    return flower_type

#def make_train_data(flower_type,DIR):
    #for img in tqdm(os.listdir(DIR)):
    #for img in tqdm(os.listdir(DIR)):
        #label=assign_label(img,flower_type)
        #path = os.path.join(DIR,img)
        #img = cv2.imread(path,cv2.IMREAD_COLOR) este
        #img = io.imread(path)
        #img = Image.open(path)
        #img = io.imread(path)
        #img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  este
        #img = img.resize((IMG_SIZE, IMG_SIZE))
        #img = img.convert('RGB')
        #img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        #img = img.convert('RGB')
        #X.append(np.array(img))
        #Z.append(str(label))

 st.write("Se agregan las imágenes a una tupla con su respectivo tipo de flor en otra tupla.")

 st.code("""
   make_train_data('Daisy',FLOWER_DAISY_DIR)
   make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
   make_train_data('Tulip',FLOWER_TULIP_DIR)
   make_train_data('Dandelion',FLOWER_DANDI_DIR)
   make_train_data('Rose',FLOWER_ROSE_DIR)
 """)

 with st.spinner("Cargando los datos, espere un momento..."):
    #make_train_data('Daisy',FLOWER_DAISY_DIR)
    #daisy_n = len(X)

    #make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
    #sunflower_n = len(X) - daisy_n

    #make_train_data('Tulip',FLOWER_TULIP_DIR)
    #tulip_n = len(X) - sunflower_n - daisy_n

    #make_train_data('Dandelion',FLOWER_DANDI_DIR)
    #dandelion_n = len(X) - tulip_n - sunflower_n - daisy_n

    #make_train_data('Rose',FLOWER_ROSE_DIR)
    #rose_n = len(X) - dandelion_n - tulip_n - sunflower_n - daisy_n

    st.write("La cantidad de imágenes de cada tipo de flores son: ")

    st.write(f"* **Margarita** (Daisy): 764")
    st.write(f"* **Girasol** (Sunflower): 733")
    st.write(f"* **Tulipán** (Tulip): 984")
    st.write(f"* **Diente de león** (Dandelion): 1052")
    st.write(f"* **Rosa** (Rose): 784")
    st.write("Se pueden ver algunas imágenes con su respectivo tipo en la siguiente gráfica:")

 st.code("""
    fig,ax=plt.subplots(3,4)
    fig.set_size_inches(15,15)
    for i in range(3):
      for j in range (4):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
    plt.tight_layout()
    plt.show()
 """)
 flores1=Image.open("flores1.png")
 st.write(flores1)

#descomentar
#fig,ax=plt.subplots(3,4)
#fig.set_size_inches(15,15)
#for i in range(3):
#    for j in range (4):
#        l=rn.randint(0,len(Z))
#        ax[i,j].imshow(X[l])
#        ax[i,j].set_title('Flower: '+Z[l])
        
#plt.tight_layout()
#st.pyplot(fig)

 st.write("Para poder ocupar las categorías se deben codificar y separando en datos de entrenamiento y prueba: ")

 st.code("""
   le=LabelEncoder()
   Y=le.fit_transform(Z)
   Y=to_categorical(Y,5)
   X=np.array(X)
   X=X/255
   x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
 """)

#le=LabelEncoder()
#Y=le.fit_transform(Z)
#Y=to_categorical(Y,5)
#X=np.array(X)
#X=X/255
#x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

#st.write(len(x_train), len(x_test), len(y_train), len(y_test))
 st.write(3237, 1080, 3237, 1080)

 st.subheader("Creación del modelo CNN")

 st.code("""
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(5, activation = "softmax"))
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    batch_size=128
    epochs=50
  """)

 model = Sequential()
 model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
 model.add(MaxPooling2D(pool_size=(2,2)))


 model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
 model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

 model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
 model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
 model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 model.add(Flatten())
 model.add(Dense(512))
 model.add(Activation('relu'))
 model.add(Dense(5, activation = "softmax"))
 batch_size=128
 epochs=50
 model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


 st.write("Además se necesita generar datos correctos para que el modelo logre generalizar lo más posibles todos los casos, para ello se hace un ImageDataGenerator, que aplica los siguientes procesos una rotación aleatorio, zoom aleatorios, desplazamientos verticales y horizontales aleatorios, reflexión horizontal.")

 st.code("""
    from keras.callbacks import ReduceLROnPlateau
    red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        horizontal_flip=True,  
        vertical_flip=False)  
    datagen.fit(x_train)
 """)

#from tensorflow.keras.callbacks import ReduceLROnPlateau
#red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

#datagen = ImageDataGenerator(
#        featurewise_center=False,
#        samplewise_center=False, 
#        featurewise_std_normalization=False,
#        samplewise_std_normalization=False, 
#        zca_whitening=False, 
#        rotation_range=10, 
#        zoom_range = 0.1, 
#        width_shift_range=0.2,
#        height_shift_range=0.2,
#        horizontal_flip=True,
#        vertical_flip=False)


#datagen.fit(x_train)


#epocas = st.radio("**Selecciones la cantidad de epocas para el entrenamiento: (3 por defecto)** ", ["3 Epocas", "50 Epocas", "100 Epocas"], captions=["1 min", "10 min", "20 min"])

#if epocas == "3 Epocas":
#    epochs=1
#    with st.spinner("El modelo se esta entrenando, espere un momento..."):
#          History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
 #                             epochs = epochs, validation_data = (x_test,y_test),
 #                             verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
#if epocas == "50 Epocas":
#    epochs=50
#    with st.spinner("El modelo se esta entrenando, espere un momento..."):
#          History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
#                              epochs = epochs, validation_data = (x_test,y_test),
#                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
#if epocas == "100 Epocas":
#    epochs=100
#    with st.spinner("El modelo se esta entrenando, espere un momento..."):
#          History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
#                              epochs = epochs, validation_data = (x_test,y_test),
#                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
    
#History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
#                              epochs = epochs, validation_data = (x_test,y_test),
#                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

 st.write("Rendimiento del modelo:")

 flores2_err=Image.open("flores2_err.png")
 flores2_acc=Image.open("flores2_acc.png")




 left, right = st.columns(2)

 with left:
   st.write(flores2_err)
   #fig,ax=plt.subplots()
   #ax.plot(History.history['loss'])
   #ax.plot(History.history['val_loss'])
   #plt.title('Función de costo del modelo')
   #plt.ylabel('Error')
   #plt.xlabel('Epocas (desde 0)')
   #plt.legend(['train', 'test'])
   #st.pyplot(fig)
 
 with right:
   st.write(flores2_acc)
   #fig,ax=plt.subplots()
   #ax.plot(History.history['accuracy'])
   #ax.plot(History.history['val_accuracy'])
   #plt.title('Puntuación del modelo')
   #plt.ylabel('Accuracy')
   #plt.xlabel('Epocas (desde 0)')
   #plt.legend(['train', 'test'])
   #st.pyplot(fig)


 st.subheader("-Predicciones")

 st.write("Se obtienen las predicciones y al mismo tiempo, se ordenan en las que fueron predichas correctamente y las que el modelo fallo.")


 st.code("""
    pred=model.predict(x_test)
    pred_digits=np.argmax(pred,axis=1)
    i=0
    prop_class=[]
    mis_class=[]
    for i in range(len(y_test)):
       if(np.argmax(y_test[i])==pred_digits[i]):
          prop_class.append(i)
       if(len(prop_class)==8):
          break
    i=0
    for i in range(len(y_test)):
       if(not np.argmax(y_test[i])==pred_digits[i]):
          mis_class.append(i)
       if(len(mis_class)==8):
          break
 """)


#pred=model.predict(x_test)
#pred_digits=np.argmax(pred,axis=1)

#i=0
#prop_class=[]
#mis_class=[]



#for i in range(len(y_test)):
#    if(np.argmax(y_test[i])==pred_digits[i]):
#        prop_class.append(i)
    #if(len(prop_class)==8):
    #    break

#i=0
#for i in range(len(y_test)):
#    if(not np.argmax(y_test[i])==pred_digits[i]):
#        mis_class.append(i)
    #if(len(mis_class)==8):
    #    break

 warnings.filterwarnings('always')
 warnings.filterwarnings('ignore')

#correct = len(prop_class)



 st.write(f"* El modelo predijo correctamente :green[780]")

 flores3=Image.open("flores3.png")
 st.write(flores3)

#count=0
#fig,ax=plt.subplots(2,4)
#fig.set_size_inches(15,15)
#for i in range (2):
#    for j in range (4):
#        ax[i,j].imshow(x_test[prop_class[count]])
        
#        predicted_label = le.inverse_transform([pred_digits[prop_class[count]]])[0]

# En lugar de envolver `y_test[prop_class[count]]` en una lista, pásalo directamente a `np.argmax`
#        actual_class = np.argmax(y_test[prop_class[count]])
#        actual_label = le.inverse_transform([actual_class])[0]

#        ax[i, j].set_title(f"Predicted Flower: {predicted_label}\nActual Flower: {actual_label}")


#        plt.tight_layout()
        
#        count+=1
#st.pyplot(fig)

#warnings.filterwarnings('always')
#warnings.filterwarnings('ignore')

#fail = len(mis_class)

 st.write(f"* El modelo predijo erroneamente :green[300]")

 flores4=Image.open("flores4.png")
 st.write(flores4)

#count=0
#fig,ax=plt.subplots(2,4)
#fig.set_size_inches(15,15)
#for i in range (2):
#    for j in range (4):
#        ax[i,j].imshow(x_test[mis_class[count]])
 
        # Obtener la etiqueta predicha
#        predicted_label = le.inverse_transform([pred_digits[mis_class[count]]])[0]

# Obtener la etiqueta real usando np.argmax directamente sobre el valor en y_test
#        actual_class = np.argmax(y_test[mis_class[count]])
#        actual_label = le.inverse_transform([actual_class])[0]

# Establecer el título del subplot con la etiqueta predicha y la etiqueta real
#        ax[i, j].set_title(f"Predicted Flower: {predicted_label}\nActual Flower: {actual_label}")

#        plt.tight_layout()
#        count+=1
#st.pyplot(fig)


 st.subheader("Conclusiones")

 st.write("El modelo de CNN hace un buen desempeño, el desempeño obtenido es de 72% de acierto y 27% de fallas, este desempeño es variable dependiendo de las imágenes de entrenamiento y prueba que tome, logrando sacar 80% de  aciertos y 20% de fallas. El tipo de flores entrenados tiene cumplen que tienen morfologías muy reconocibles entre ellas por lo que debería hacer un buen desempeño, pero puede no ser el esperado si fueran más parecidas. El modelo sobre 60 épocas tiene a aumentar el error, por lo que lo más óptimo es cercano a 50 épocas.")





























































