import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

import base64
import streamlit as st



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

#from yellowbrick.cluster import KElbowVisualizer

#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn

def show_page():

 @st.cache_data
 def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

 Codo=Image.open("Codo1.png")
 Dash1=Image.open("dashboard1.png")
 df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv', index_col=0)

 st.title("Segmentación de clientes")
 st.write(":green[*Modelo ML - Clustering*]")
 st.write("Un clustering corresponde a un análisis para llevar a cabo una segmentación, al hacerse un análisis generalmente se da como resultado que una sola segmentación es la más óptima, por lo que no puede ser interactivo al ser solo un resultado, por esta razón esta sección será un análisis para lograr definir las diferentes agrupaciones de clientes.")
#print(df)

 st.write("Estos datos corresponden a los clientes de una empresa, en los cuales se puede ver que se encuentran columnas como ingresos anuales, puntaje de gasto, edad y género de los clientes. Se puede descargar los datos dando click [aquí](%s)." % "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv")

 st.write("Comúnmente en los datos de las empresas se encuentran transacción por transacción de los clientes, por lo que en ese caso se puede ocupar groupby para lograr obtener datos como estos.")

 st.code("df.head(5)")

 st.table(df.head(5))

 st.write("Nos damos cuenta de que hay un posible problema porque el índice no empieza de 0.")

 df=df.reset_index(drop=True)           

 st.code("df.reset_index(drop=True)")

 
 from sklearn.preprocessing import LabelEncoder

 st.write("Codificamos la columna que contiene los géneros para poder ocuparla.")

 le = LabelEncoder()
 data = le.fit_transform(df["Gender"])
 data = pd.DataFrame(data, columns=["Gender"])
 df=df.drop(columns=["Gender"])
 df = pd.concat([df, data], axis=1, sort=True)


 st.code("""
       le = LabelEncoder()
       data = le.fit_transform(df["Gender"])
       data = pd.DataFrame(data, columns=["Gender"])
       df=df.drop(columns=["Gender"])
       df = pd.concat([df, data], axis=1, sort=True)
 """)

 st.table(df.head(5))

 st.write("En este caso ocupamos KMeans es el que da mejor resultado y se procede a saber cuanto son los clusters óptimos que podría tener el modelo.")


 st.code("""
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=10)
    visualizer.fit(df)        
    visualizer.show()   
 """)

 st.write(Codo)

 st.write("La gráfica anterior muestra que el número óptimo de clusters es 5, ya que se produce un codo en la puntuación de distorsión, por lo que se procederá a entrenar un modelo que clasifique 5 clusters.")

 km = KMeans(n_clusters=5, random_state=42, n_init=10)
 km.fit(df)

 st.code("""
   km = KMeans(n_clusters=5, random_state=42, n_init=10)
   km.fit(df)
 """)

 st.write("Luego de ajustar con el modelo KMeans podemos ver los resultados.")

 st.code("""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection="3d") 
    ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], marker="o", c=km.labels_, s=150)
    ax.set_title("Proyeccion 3d de los datos con dimensionalidad reducida")
    ax.set_xlabel("Ingresos anuales")
    ax.set_ylabel("Puntuación de gasto")
    ax.set_zlabel("Edad")
    plt.show()
 """)




 fig = plt.figure(figsize=(8,8))
 ax = fig.add_subplot(111, projection="3d")
 ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], marker="o", c=km.labels_, s=150)
 ax.set_title("Proyeccion 3d de los datos")
 ax.set_xlabel("Ingresos anuales k$")
 ax.set_ylabel("Puntuación de gasto (1-100)")
 ax.set_zlabel("Edad")
 st.pyplot(fig)

 st.write("Se puede notar que el modelo no tomo en cuenta la edad, le dio poca importancia, porque se ve que el eje z cubre todo el rango de valores sin segmentación. Las columnas más importantes para la segmentación son los ingresos anuales y la puntuación de gasto.")

 left, right = st.columns(2)
 
 with left:
   fig, ax = plt.subplots(figsize=(8,8))
   ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=km.labels_, marker="o", s=150)
   ax.set_xlabel("Ingreso anual (k$)")
   ax.set_ylabel("Puntuación de gasto (1-100)")
   st.pyplot(fig)
   
 with right:
   fig, ax = plt.subplots(figsize=(8,8))
   ax.scatter(df['Age'], df['Spending Score (1-100)'], c=km.labels_, marker="o", s=150)
   ax.set_xlabel("Edad")
   ax.set_ylabel("Puntuación de gasto (1-100)")
   st.pyplot(fig)
#fig = px.scatter_3d(df['Annual Income (k$)'], df['Spending Score (1-100)'], df["Age"], c=km.labels_)
   

 print(df.shape)

 st.write("Con los clusters formados agregamos esta nueva clasificación a los datos para explorarlos posteriormente.")

 kmc = pd.DataFrame(km.labels_.astype("int"), columns=["cluster KM"])
 df = pd.concat([df, kmc], axis=1, sort=True)

 st.code("""
    kmc = pd.DataFrame(km.labels_.astype("int"), columns=["cluster KM"])
    df = pd.concat([df, kmc], axis=1, sort=True)
    df.head(5)
 """)

 st.table(df.head(5))
#st.table(df.groupby(["cluster KM"]).mean())
 st.write("Era datos aparentemente simples, pero gracias a segmentación de clientes se puede ver que los clientes características muy similares entre ellas podemos ver:")
 st.write("* El gasto no tiene que ver con los género, porque tienen gastos similares tanto hombres como mujeres, sino mas bien con el tipo de clientes que son dependiendo de sus ingresos.")
 st.write("* El gasto en los clusters es muy similar entre clientes que pertenecen al mismo cluster, el modelo los asocio según sus ingresos y gastos, encontró que compartían características similares.")
 st.write("* Si bien la edad no están relevante para segmentar clientes en este caso, si hay ciertos rangos etarios, que son más seguros en la compra de la empresa como clientes entre 20-29 y 30-39, esto podría darnos información para donde dirigir las publicidades y promociones, puede ser útil cuando se necesite captar clientes o reafirmar la decisión de compra de los clientes de esas edades.")

 st.write(Dash1)

 st.write("En resumen el modelo separó los clientes en clusters que cumplen las siguientes características:")

 st.write("* **Cluster 0:** Tiene bajos ingresos y bajos gastos en la empresa.")
 st.write("* **Cluster 1:** Tiene ingresos medios y gastos medios en la empresa, es el tipo de cliente más importante para la empresa porque es el más numeroso.")
 st.write("* **Cluster 2:** Tiene altos ingresos y altos gastos en la empresa.")
 st.write("* **Cluster 3:** Tiene altos ingresos y bajos gastos en la empresa.")
 st.write("* **Cluster 4:** Tiene bajos ingresos y altos gastos en la empresa.")



