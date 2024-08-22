import streamlit as st
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

 @st.cache_data
 def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

 data = pd.read_csv('netflix_titles.csv')

 st.title("Sistema de recomendación")

 st.write(":green[*Modelo ML - Vectorización*]")

 st.subheader("Exploración y Análisis")

 st.write("Los datos corresponden tanto a series como películas de netflix, en la cual frente a diferentes selecciones del usuario, se deben mostrar recomendaciones similares. El sistema de recomendación lo hace por mediante similitud de palabras, por lo que es posible aplicarlo a cualquier colección de datos que tenga palabras.")

 st.write("Un sistema de recomendación es un problema que requiere una estrategia, esta estrategia se trata de elegir las características más importante del producto para realizar la recomendación, puede depender de en que situación lo queremos aplicar, por ejemplo un usuario quiere un tema en particular, eso será prioridad o puede que le interese más el título de la película, serán recomendaciones diferentes.")

 st.write("En este caso la estrategia es enfocada en el nombre del título y la trama")

 st.subheader("Manipulación y limpieza")

 st.write("Los datos se presentan a continuación:")

 st.code("""
    data.head(5)
    data.shape
    """)

 st.write(data.head(5))
 st.write(data.shape)

 st.write("Vemos cuantos datos nulos existen en la data por columnas:")

 st.code("""
    pd.DataFrame(data.isna().sum()).T
    """)

 st.write(pd.DataFrame(data.isna().sum()).T)

 st.write("Existe muchos nulos por lo que no es recomendables ocupar esas columnas, por lo que en este proyecto se estudiara cual modelo es mejor uno que tome columnas diferentes, siempre evitando ocuparlas.")

 st.write("Las columnas 'country', 'date_added', 'rating', 'duration' serán remplazados por la moda, ya que no es tan necesario que tengan el dato real todos, no importa si es que tiene pequeñas equivocaciones.")


 st.code("""
    data["country"]=data["country"].fillna(data["country"].mode()[0])
    data["date_added"]=data["date_added"].fillna(data["date_added"].mode()[0])
    data["rating"] = data["rating"].fillna(data["rating"].mode()[0])
    data["duration"] = data["duration"].fillna("90 min")  
    pd.DataFrame(data.isna().sum()).T
    """)
 data["country"]=data["country"].fillna(data["country"].mode()[0])
 data["date_added"]=data["date_added"].fillna(data["date_added"].mode()[0])
 data["rating"] = data["rating"].fillna(data["rating"].mode()[0])
 data["duration"] = data["duration"].fillna("90 min")  

 data_copy = data.copy()

 st.write(pd.DataFrame(data.isna().sum()).T)

 st.write("Para las columnas 'director' y 'cast', se verá más adelante si es que se pueden ocupar, dependiendo si es necesario ocuparlas, para ocuparlas se debe reducir la data inevitablemente.")

 st.subheader("-Ingeniería de características")

 st.write("Primero aplicar algunos cambios para poder explorar mejor los datos como agregar una columna solo son los años de la fecha de publicación, quedarse solo con una clasificación la primera de la producción y el país principal de creación de la producción. ")

 st.code("""
     data["listed_in"]=data["listed_in"].apply(lambda x: x.split(",")[0])
     data["year_add"]=data["date_added"].apply(lambda x: x.split(" ")[-1])
     data["country_main"]=data["country"].apply(lambda x: x.split(",")[0])
   """)

 data["listed_in"]=data["listed_in"].apply(lambda x: x.split(",")[0])
 data["year_add"]=data["date_added"].apply(lambda x: x.split(" ")[-1])
 data["country_main"]=data["country"].apply(lambda x: x.split(",")[0])

 st.write(data.head(5))

 st.write("Revisando primero las producciones y como estas se distribuyen en los años.")

 st.code("""
      data2=pd.DataFrame(data.groupby(["year_add"], as_index=False)["duration"].agg("count"))
      fig, ax=plt.subplots()
      sns.set(style="darkgrid")
      sns.lineplot(x=data2["year_add"], y=data2["duration"])
      plt.ylabel("Cantidad de producciones")
      plt.xlabel("Año de publicación")
      plt.title("Producciones publicadas por año en netflix")
      plt.xticks(rotation=45)
      plt.show() 
   """)

 data2=pd.DataFrame(data.groupby(["year_add"], as_index=False)["duration"].agg("count"))
 fig, ax=plt.subplots()
 sns.set(style="darkgrid")
 sns.lineplot(x=data2["year_add"], y=data2["duration"])
 plt.ylabel("Cantidad de producciones")
 plt.xlabel("Año de publicación")
 plt.title("Producciones publicadas por año en netflix")
 plt.xticks(rotation=45)
 st.pyplot(fig)

 st.write("Vemos que los datos tienen producciones publicadas desde el año 2008 - 2021 donde la mayoría de las publicaciones de producciones se dieron entre 2015 - 2021, siendo muy pocas las que se publicaron antes.")

 st.write("Vemos cuales son los principales países que producen estas producciones:")

 st.code("""
     data3=pd.DataFrame(data.groupby(["country_main"], as_index=True)["country_main"].agg("count"))
     data3=data3[data3["country_main"]>100]
     fig, ax=plt.subplots()
     sns.set(style="darkgrid")
     plt.bar(data3.index, data3["country_main"])
     plt.ylabel("Cantidad de publicaciones")
     plt.title("Paises con más de 100 publicaciones en netflix")
     plt.xticks(rotation=80)
     plt.show()
     """)

 data3=pd.DataFrame(data.groupby(["country_main"], as_index=True)["country_main"].agg("count"))
 data3=data3[data3["country_main"]>100]
 fig, ax=plt.subplots()
 sns.set(style="darkgrid")
 plt.bar(data3.index, data3["country_main"])
 plt.ylabel("Cantidad de publicaciones")
 plt.title("Paises con más de 100 publicaciones en netflix")
 plt.xticks(rotation=80)
 st.pyplot(fig)

 st.write("Se muestra que estados unidos que tiene más publicaciones en netflix con una gran mayoría, recordando que 831 se le agregaron adicionales, pero son datos ficticios, aun así tiene una amplia mayoría de las publicaciones, seguido por india y reino unido.")

 st.write("Otro de los aspectos importantes es ver que tipo de contenido hay en netflix: ")

 st.code("""
   data3=pd.DataFrame(data.groupby(["listed_in"], as_index=True)["listed_in"].agg("count"))
   data3=data3[data3["listed_in"]>100]
   fig, ax=plt.subplots()
   sns.set(style="darkgrid")
   cmap = plt.get_cmap('tab20')
   plt.bar(data3.index, data3["listed_in"], color=[cmap(i) for i in np.linspace(0, 1, len(data3["listed_in"]))])
   plt.ylabel("Cantidad de publicaciones")
   plt.title("Categorias con mas de 100 publicaciones en netflix")
   plt.xticks(rotation=80)
   plt.show()
   """)

 data3=pd.DataFrame(data.groupby(["listed_in"], as_index=True)["listed_in"].agg("count"))
 data3=data3[data3["listed_in"]>100]
 fig, ax=plt.subplots() 
 sns.set(style="darkgrid")
 cmap = plt.get_cmap('tab20')
 plt.bar(data3.index, data3["listed_in"], color=[cmap(i) for i in np.linspace(0, 1, len(data3["listed_in"]))])
 plt.ylabel("Cantidad de publicaciones")
 plt.title("Categorias con mas de 100 publicaciones en netflix")
 plt.xticks(rotation=80)
 st.pyplot(fig)

 st.write("La mayoría de las producciones son 'Dramas', seguido de 'Comedias' y 'Acción y aventuras' es importante recordar que solo son consideradas una de todas las posibles categorías de cada producción, solo se muestran las que tienen sobre 100 producciones.")

 st.write("En cuanto a tipo de producciones se distribuyen de la siguiente manera:")


 st.code("""
    count = data["type"].value_counts()
    percent=100*data["type"].value_counts(normalize=True)
    resumen1 = pd.DataFrame({"count":count, "percent": percent.round(1)})
    st.write(resumen1)
    fig, ax=plt.subplots()
    plt.pie(x=resumen1["percent"], labels=resumen1.index, autopct='%1.1f%%')
    plt.title("Tipo de producciones de netflix")
    for i, texto in enumerate(plt.gca().texts):
      texto.set_rotation(45)
    plt.show()
   """)

 count = data["type"].value_counts()
 percent=100*data["type"].value_counts(normalize=True)
 resumen1 = pd.DataFrame({"count":count, "percent": percent.round(1)})


#st.write(resumen1)
 fig, ax=plt.subplots()
 plt.pie(x=resumen1["percent"], labels=resumen1.index, autopct='%1.1f%%')
 plt.title("Tipo de producciones de netflix")
 for i, texto in enumerate(plt.gca().texts):
    texto.set_rotation(45)
 st.pyplot(fig)

 st.write("Son mucho más las películas en cantidad en comparación a los show televisivos, en los show televisivos se encuentran todas las series, pero este gráfico es engaño, puesto que las series se considera solo 'una', pero son muchos episodios por lo tanto en cantidad de contenido pueden ser similares.")

 st.write("Ahora es importante ver como se distribuyen las clasificación del contenido en cuanto a que tipo de personas está dirigido:")
 data_mod = data.copy()

#st.write(data_mod)

 data_mod["rating"] = data_mod["rating"].replace({"PG":"Otros", "TV-Y7":"Otros", "TV-Y":"Otros", "NR":"Otros", "TV-G":"Otros", "TV-Y7-FV":"Otros", "NC-17":"Otros", "74 min":"Otros", "84  min":"Otros", "66 min":"Otros", "84 min":"Otros", "UR":"Otros", "G":"Otros"})

 st.code("""
   count = data_mod["rating"].value_counts()
   percent=100*data_mod["rating"].value_counts(normalize=True)
   resumen2 = pd.DataFrame({"count":count, "percent": percent.round(1)}
   fig, ax=plt.subplots()
   plt.pie(x=resumen2["percent"], labels=resumen2.index, autopct='%1.1f%%')
   plt.title("Clasificación de contenido en netflix")
   for i, texto in enumerate(plt.gca().texts):
      texto.set_rotation(0)
   plt.show()
 """)
 count = data_mod["rating"].value_counts()
 percent=100*data_mod["rating"].value_counts(normalize=True)
 resumen2 = pd.DataFrame({"count":count, "percent": percent.round(1)})

 fig, ax=plt.subplots()
 plt.pie(x=resumen2["percent"], labels=resumen2.index, autopct='%1.1f%%')
 plt.title("Clasificación de contenido en netflix")
 for i, texto in enumerate(plt.gca().texts):
    texto.set_rotation(0)
 st.pyplot(fig)

 st.write("Se ve que la clasificación 'TV-MA' es la predominante en netflix, se trata de una clasificación para adultos, seguido de 'TV-14' con supervisión de padres, mientras que otras clasificaciones son minoría, más de la mitad del contenido es considerada para adultos o con supervisión de adultos.")

 st.subheader("-Selección de método de recomendación")

 st.write("Se puede intentar primero el resultado con la descripción de las producciones que tiene que ver respecto a la trama :red[TfidfVectorizer] relacionadas con :red[linear_kernel]:")
#data = data.drop(['director', 'cast'], axis=1)


 st.code("""
     from sklearn.feature_extraction.text import TfidfVectorizer
     tf_2 = TfidfVectorizer(stop_words="english")
     data["description"] = data["description"].fillna("")
     tf_2_matrix = tf_2.fit_transform(data["description"])
     st.write(tf_2_matrix.head(5))
     st.write(tf_2_matrix.shape)
 """)
 from sklearn.feature_extraction.text import TfidfVectorizer

 tf_2 = TfidfVectorizer(stop_words="english")
 data["title"] = data["title"].fillna("")
 tf_2_matrix = tf_2.fit_transform(data["title"])

 st.write(tf_2_matrix.shape)

 st.write("A continuación se crea la matriz y luego se hace la recomendaciones")

 st.code("""
   from sklearn.metrics.pairwise import linear_kernel
   cosine_sim = linear_kernel(tf_2_matrix, tf_2_matrix)
   indices = pd.Series(data.index, index=data["title"]).drop_duplicates()
   #funcion que hara las recomendaciones
   def get_recommendation(title, cosine_sim=cosine_sim):
         idx=indices[title]
         sim_scores = list(enumerate(cosine_sim[idx]))
         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
         sim_scores = sim_scores[1:21]      
         movie_indices = [i[0] for i in sim_scores]
         return data["title"].iloc[movie_indices]
   """)

 st.write("Se hacen dos recomendaciones en simultáneas para ver diferentes ejemplos: ")

 from sklearn.metrics.pairwise import linear_kernel

 cosine_sim = linear_kernel(tf_2_matrix, tf_2_matrix)

 indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

#st.write(indices)

 def get_recommendation(title, cosine_sim=cosine_sim):
      idx=indices[title]
      #obtenemos puntuciones segun similitud
      sim_scores = list(enumerate(cosine_sim[idx]))
      #ordenamos segun su puntuacion
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      #que muestre solo las 10 primeras
      sim_scores = sim_scores[1:21]      
      movie_indices = [i[0] for i in sim_scores]
      
      #devuelve el top 10
      return st.write(data["title"][data["title"]==title]), st.write(data["title"].iloc[movie_indices])

#apareceran las puntuaciones con esto 


 st.subheader("Recomendación por una característica")

 st.write("* Recomendación solo por el nombre del :red[**Título**]:")
 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   get_recommendation("Naruto")

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   get_recommendation("The Boy")
#puede que no nos guste asi que le agregamos mas cosas

 tf_2 = TfidfVectorizer(stop_words="english")
 data["description"] = data["description"].fillna("")
 tf_2_matrix = tf_2.fit_transform(data["description"])



 from sklearn.metrics.pairwise import linear_kernel

 cosine_sim = linear_kernel(tf_2_matrix, tf_2_matrix)

 indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

#st.write(indices)

 def get_recommendation(title, cosine_sim=cosine_sim):
      idx=indices[title]
      #obtenemos puntuciones segun similitud
      sim_scores = list(enumerate(cosine_sim[idx]))
      #ordenamos segun su puntuacion
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      #que muestre solo las 10 primeras
      sim_scores = sim_scores[1:21]      
      movie_indices = [i[0] for i in sim_scores]
      
      #devuelve el top 10
      return st.write(data["title"][data["title"]==title]), st.write(data["title"].iloc[movie_indices])


 st.write("* Recomendación solo con la :red[**Descripción**]:")
 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   get_recommendation("Naruto")

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   get_recommendation("The Boy")



 tf_2 = TfidfVectorizer(stop_words="english")
 data["rating"] = data["rating"].fillna("")
 tf_2_matrix = tf_2.fit_transform(data["rating"])



 from sklearn.metrics.pairwise import linear_kernel

 cosine_sim = linear_kernel(tf_2_matrix, tf_2_matrix)

 indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

#st.write(indices)

 def get_recommendation(title, cosine_sim=cosine_sim):
      idx=indices[title]
      #obtenemos puntuciones segun similitud
      sim_scores = list(enumerate(cosine_sim[idx]))
      #ordenamos segun su puntuacion
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      #que muestre solo las 10 primeras
      sim_scores = sim_scores[1:21]      
      movie_indices = [i[0] for i in sim_scores]
      
      #devuelve el top 10
      return st.write(data["title"][data["title"]==title]), st.write(data["title"].iloc[movie_indices])


 st.write("* Recomendación solo con la :red[**Clasificación de contenido**]:")
 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   get_recommendation("Naruto")

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   get_recommendation("The Boy")

 st.write("En general la recomendación con 'Título' hace un buen desempeño cuando se busca algo en particular o si es que el contenido tiene el mismo nombre, en cuanto a 'Descripción' hace buen papel para recomendar contenido adicional, pero no es perfecto por lo que necesita una prueba más, en la caso de 'Clasificación de contenido' puede llegar a tener buenas recomendaciones, pero mezcla otras producciones que no tienen anda que ver.")


 from sklearn.feature_extraction.text import CountVectorizer

 tf_2 = CountVectorizer(stop_words="english")
 data["description"] = data["description"].fillna("")
 tf_2_matrix = tf_2.fit_transform(data["description"])



 from sklearn.metrics.pairwise import linear_kernel

 cosine_sim = linear_kernel(tf_2_matrix, tf_2_matrix)

 indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

#st.write(indices)

 def get_recommendation(title, cosine_sim=cosine_sim):
      idx=indices[title]
      #obtenemos puntuciones segun similitud
      sim_scores = list(enumerate(cosine_sim[idx]))
      #ordenamos segun su puntuacion
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      #que muestre solo las 10 primeras
      sim_scores = sim_scores[1:21]      
      movie_indices = [i[0] for i in sim_scores]
      
      #devuelve el top 10
      return st.write(data["title"][data["title"]==title]), st.write(data["title"].iloc[movie_indices])


 st.write("* Recomendación solo con la :red[**Descripción con CountVectorizer**]:")
 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   get_recommendation("Naruto")

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   get_recommendation("The Boy")


 st.write("En general es bastante parecido a las recomendaciones ocupando TfidfVecorizer, no muestras grandes diferencias.")

 st.subheader("Recomendaciones con múltiples características")

 st.write("* Caso con todas las columnas que se consideren importantes, :red[Sin 'director' y 'cast']:")

 st.code("""
     features = ["title", "description", "rating", "type", "listed_in", "country"]
     filtros = data_copy[features]
     def clean_data(x):
          return str.lower(x.replace(" ", ""))
     for feature in features:
          filtros[feature] = filtros[feature].apply(clean_data)
     def create_soup(x):
          return x["title"] + ","+ x["description"] + ","+ x["rating"] + "," + x["listed_in"] + "," + x["country"] + "," + x["type"]
     filtros["soup"] = filtros.apply(create_soup, axis=1)
     from sklearn.feature_extraction.text import TfidfVectorizer
     count = TfidfVectorizer(stop_words='english')
     count_matrix = count.fit_transform(filtros['soup'])
     from sklearn.metrics.pairwise import cosine_similarity
     cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
     filtros = filtros.reset_index()
     indices = pd.Series(filtros.index, index=filtros["title"])
     def get_recommendations_new(title, cosine_sim=cosine_sim2):
           title=title.replace(" ", "").lower()
           idx = indices[title]           
           sim_scores=list(enumerate(cosine_sim[idx]))
           sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
           sim_scores=sim_scores[1:21]
           movie_indices = [i[0] for i in sim_scores]
           return data["title"].iloc[movie_indices] 
   """)

 features = ["title", "description", "rating", "type", "listed_in", "country", "duration"] #"description"]
#features = ["title", "description", "listed_in"]
 filtros = data_copy[features]

 def clean_data(x):
          return str.lower(x.replace(" ", ""))
          
 for feature in features:
       filtros[feature] = filtros[feature].apply(clean_data)
       
      
 def create_soup(x):
       return x["title"] + ","+ x["description"] + ","+ x["rating"] + "," + x["listed_in"] + "," + x["country"] + "," + x["type"] + "," + x["duration"]

#def create_soup(x):
#       return x["title"] + ","+ x["description"] + x["listed_in"]

 filtros["soup"] = filtros.apply(create_soup, axis=1)

#esto es lo mismo que lo anterior pero con countvectorizer
 from sklearn.feature_extraction.text import TfidfVectorizer


 count = TfidfVectorizer(stop_words='english')
 count_matrix = count.fit_transform(filtros['soup'])

 from sklearn.metrics.pairwise import cosine_similarity

 cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#st.write(filtros.head(5))

 filtros = filtros.reset_index()
 indices = pd.Series(filtros.index, index=filtros["title"])

 def get_recommendations_new(title, cosine_sim=cosine_sim2):
           title=title.replace(" ", "").lower()
           idx = indices[title]
           
           sim_scores=list(enumerate(cosine_sim[idx]))
           sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
           sim_scores=sim_scores[1:21]
           movie_indices = [i[0] for i in sim_scores]
           return data["title"].iloc[movie_indices]           
           

 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   st.write(get_recommendations_new("Naruto"))

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   st.write(get_recommendations_new("The Boy"))

 st.write("Al hacer una sopa de palabras, al poder incorporar las columnas países y la categoría de la producción que en principio no se podía ocupar al tener producciones con muchas categorías y otras pocas, ahora en la sopa hace un buen entendimiento y logra recomendar películas muy parecidas, también del mismo género, pero en el caso de 'Naruto' no logra mostrar las propuestas directas que son las otras series de 'Naruto'.")

 st.write("* Recomendaciones :red[incluyendo 'director' y 'cast'], y por lo tanto eliminando muchos de los datos nulos:")

 features = ["title", "director", "cast", "description", "rating", "type", "listed_in", "country", "duration"] #"description"]
#features = ["title", "description", "listed_in"]
 data_copy = data_copy.dropna(subset=["director", "cast"])
 st.write(data_copy.shape)
 filtros = data_copy[features]

 def clean_data(x):
          return str.lower(x.replace(" ", ""))
          
 for feature in features:
       filtros[feature] = filtros[feature].apply(clean_data)
       
      
 def create_soup(x):
       return x["title"] + ","+ x["description"] + ","+ x["rating"] + "," + x["listed_in"] + "," + x["country"] + "," + x["type"] + "," + x["director"] + "," + x["cast"] + "," + x["duration"]

#def create_soup(x):
#       return x["title"] + ","+ x["description"] + x["listed_in"]

 filtros["soup"] = filtros.apply(create_soup, axis=1)

#esto es lo mismo que lo anterior pero con countvectorizer
 from sklearn.feature_extraction.text import TfidfVectorizer


 count = TfidfVectorizer(stop_words='english')
 count_matrix = count.fit_transform(filtros['soup'])

 from sklearn.metrics.pairwise import cosine_similarity

 cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#st.write(filtros.head(5))

 filtros = filtros.reset_index()
 indices = pd.Series(filtros.index, index=filtros["title"])

 def get_recommendations_new(title, cosine_sim=cosine_sim2):
           title=title.replace(" ", "").lower()
           idx = indices[title]
           
           sim_scores=list(enumerate(cosine_sim[idx]))
           sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
           sim_scores=sim_scores[1:21]
           movie_indices = [i[0] for i in sim_scores]
           return data["title"].iloc[movie_indices]           
           

 left, right = st.columns(2)
 with left:
   st.write("En este caso se buscará :blue['Naruto'], el cual es una anime con múltiples producciones de aventura en general.")     
   st.write(get_recommendations_new("Naruto"))

 with right:
   st.write("En este caso se buscará :blue['The Boy'] el cual es una película de terror sin escenas tan explicitas con suspenso.")
   st.write(get_recommendations_new("The Boy"))


 st.subheader("Conclusión")

 st.write("Una recomendación buena es difícil de lograr, ningún método pudo estar cerca de ello por las producciones tienen diferentes características, sin embargo si es posible hacer buenas recomendaciones con los siguientes puntos:")

 st.write("* La recomendación ideal debe ser una mezcla de estos métodos")

 st.write("* Las recomendaciones con características individuales encuentra coincidencias directas por el 'título', mientras que 'descripción' encuentra similitudes interesantes")

 st.write("* En cuanto a la recomendación múltiple es mucho mejor que la individual en trama, logra encontrar categorías audiovisuales directas, pero no relaciona ninguna característica como la más importante.")

 st.write("* El director y el casting no son características útiles por tener que eliminar datos, pero también segregan el contenido, muestran relaciones que no tienen nada que ver, no se deben ocupar.")

 st.write("* La recomendación deben ser una mezcla primero de la búsqueda por 'Título', 'descripción' y para complementar con recomendación múltiple")



















