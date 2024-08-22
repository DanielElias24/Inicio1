import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from streamlit_extras.mention import mention
import streamlit.components.v1 as components

st.set_page_config(page_title="Proyecto", page_icon="📈")
import base64
import streamlit as st
from streamlit_option_menu import option_menu
import importlib

#if 'page' not in st.session_state:
#    st.session_state['page'] = 'Inicio'

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# Función para inyectar CSS personalizado
def set_custom_css():
    st.markdown(
        """
        <style>
        /* Estilos para la barra de desplazamiento en la página */
        ::-webkit-scrollbar {
            width: 16px; /* Ajusta el ancho de la barra de desplazamiento */
            height: 16px; /* Ajusta la altura de la barra de desplazamiento (horizontal) */
        }

        ::-webkit-scrollbar-track {
            background: #f1f1f1; /* Color del fondo de la pista de desplazamiento */
        }

        ::-webkit-scrollbar-thumb {
            background: #888; /* Color de la parte deslizable de la barra */
            border-radius: 10px; /* Radio de borde de la parte deslizable */
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #555; /* Color de la parte deslizable cuando se pasa el ratón */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main1():
    set_custom_css()

    st.write(''*1000)

if __name__ == "__main__":
    main1()



#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}
img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.colorhexa.com/191b20.png");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
'''
[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
background-size: 150%;
background-position: top left; 
background-repeat: repeat;
background-attachment: fixed;
}}
'''
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


embed_component = {'Linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script> <div class="badge-base LI-profile-badge" data-locale="es_ES" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="danielchingasilva" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://cl.linkedin.com/in/danielchingasilva?trk=profile-badge">Daniel Elías Chinga Silva</a></div> """}




import streamlit as st

# Insertando CSS para colocar una barra en la parte superior
st.markdown(
    """
    <style>
    .top-bar {
        background-color: #4A90E2;
        color: white;
        padding: 1px;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        text-align: center;
    }
    .top-bar a {
            color: white;
            margin-right: 0px;
            text-decoration: none;
            font-size: 25px; /* Tamaño de fuente de los enlaces */
        }

    .main-content {
        padding-top: 60px;  /* Espacio para que no tape el contenido */
    }
    </style>
    """,
    unsafe_allow_html=True
)
#<h1>Mi Aplicación</h1>
st.markdown(
    """
    <div class="top-bar">
        #<h1></h1>
        <div>
        <a href="https://www.linkedin.com/in/danielchingasilva/" style="color:white; margin-right:20px;">Inicio</a>
        <a href="?section=Proyectos" style="color:white; margin-right:20px;">Proyectos</a>
        <a href="?section=Lenguajes" style="color:white; margin-right:20px;">Lenguajes y librerías</a>
        <a href="?section=Sobre" style="color:white; margin-right:20px;">Sobre mí</a>
    </div>
    """,
    unsafe_allow_html=True
)


# Botones ocultos para cambiar la página en el estado de Streamlit
#st.markdown('<a id="inicio" style="display:none;" ></a>', unsafe_allow_html=True)
#st.markdown('<a id="proyectos" style="display:none;" ></a>', unsafe_allow_html=True)
#st.markdown('<a id="lenguajes" style="display:none;" ></a>', unsafe_allow_html=True)
#st.markdown('<a id="sobre" style="display:none;" ></a>', unsafe_allow_html=True)




@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

#"https://images.unsplash.com/photo-1501426026826-31c667bdf23d"
#data:image/png;base64,{img}img = get_img_as_base64("image.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://html-color.org/es/FFFFFF.jpg");
background-size: 100%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
'''
[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
background-size: 150%;
background-position: top left; 
background-repeat: repeat;
background-attachment: fixed;
}}
'''
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

import streamlit as st

# CSS para el cuadro
st.markdown("""
    <style>
    .cuadro {
        border: 2px solid #333; /* Borde del cuadro */
        padding: 15px; /* Espaciado interno */
        margin: 20px; /* Espaciado externo */
        border-radius: 8px; /* Esquinas redondeadas */
        background-color: #f5f5f5; /* Color de fondo */
        width: fit-content; /* Ajusta el ancho al contenido */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Sombra sutil */
    }
    .texto-grande {
        font-size: 1.2em; /* Tamaño del texto */
        color: #333; /* Color del texto */
    }
    </style>
""", unsafe_allow_html=True)

# HTML para el cuadro




#st.subheader(":orange[Bienvenidos] 👋")
import streamlit as st

# Insertando CSS para personalizar el título con una nueva fuente y más estilos



st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

    .custom-title {
        font-family: 'Comic Sans', selif;
        font-size: 48px;
        font-weight: bold;
        color: #24211e;
        text-align: center;
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
    </style>
    <h1 class='custom-title'>Daniel Elías Chinga Silva</h1>
    """,
    unsafe_allow_html=True
 )

 #st.title("Proyectos de machine learning")

left, right = st.columns(2)

with left:
    components.html(embed_component["Linkedin"], height=370)

with right:

    st.subheader("Autor: Daniel C. S.")
    st.write("https://github.com/DanielElias23")

    st.write("www.linkedin.com/in/danielchingasilva")
     
    st.subheader("Objetivos")

    st.write("-Dar a conocer el conocimientos al respecto con machine learning")

    st.write("-Demostrar dominio de conocimiento para solucionar problemas a diferentes problemáticas")

    st.write("-Mostrar habilidades de programación enfocado al contexto ciencia de datos en las empresas")
#mention(label="DanielElias23", icon="github", url="https://github.com/DanielElias23",)



# Función para inyectar CSS personalizado
def set_custom_css():
    st.markdown(
        """
        <style>
        /* Estilo para el contenedor de los círculos */
        .circle-container {
            display: flex;
            gap: 20px; /* Espacio entre los círculos */
            justify-content: center;
            flex-wrap: wrap;
        }
        /* Estilo para los círculos */
        .circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            background-color: #007bff; /* Color de fondo por defecto */
        }
        .circle.red { background-color: #ef9c3d; } /* Rojo */
        .circle.green { background-color: #28a745; margin-top: 90px;} /* Verde */
        .circle.blue { background-color: #007bff; } /* Azul */
        .circle.yellow { background-color: #ffc107; margin-top: 90px;} /* Amarillo */
        .circle.purple { background-color: #6f42c1; } /* Morado */
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_custom_css()

    st.markdown(
        """
        <div class="circle-container">
            <div class="circle red">Python</div>
            <div class="circle green">SQL/NoSQL</div>
            <div class="circle blue">Machine learning</div>
            <div class="circle yellow">Analitica</div>
            <div class="circle purple">Progamación</div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

st.subheader("Contacto")


left1, right1 = st.columns(2)
 
with left1:
  st.write("**Email:**")
  st.text("danielchingasilva@gmail.com")
with right1:
  st.markdown("**Telefono/Watsap**")
  st.markdown("+5698306909")

 #st.sidebar.header("Barra de proyectos")
#st.sidebar.write("Selecciona un proyecto")
#st.subhe
#ader("Problema")

#st.subheader("Descripción del problema")

#st.write("""
#         Estos proyectos son con el fin de mostrar habilidades de programación enfocado al área de ciencia de datos, los datos utilizados tienen sus contextos propios por lo que los modelos de inteligencia artificial no se pueden ocupar para uso general. Cada proyecto pretende mostrar habilidades diferentes en el contexto de machine learning, usando modelos diferentes y de diferentes categorías. Estos proyecto ya han sido realizados en los ejemplos mostrados en mí página de GitHub, pero no fueron implementados para la visualización de página web. Los análisis de los datos y las decisiones como la elección de modelos de machine learning está en el código en GitHub. 
#         """)


         



#st.write("-Predecir el valor de diferentes automoviles segun sus caracterisitcas para que la empresa pueda definir un rango de precios para ofrecer")

#st.write("""El valor del precio predicho podria informarnos del rango de precio de venta puede optar la empresa y asi definir
 #         las posibles ganacias de la empresa""")


#data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv')

#st.subheader("Descripción del dataset")

#st.write("""
#         El dataset muestra la descripción automoviles de diferentes marcas con las especificaciones tecnicas
#         de cada modelo con su respectivo precio.
#         """)

#st.write(":blue[205 x 26]")

#st.write(data)


 #page_bg_img = f"""
 #<style>
 #[data-testid="stAppViewContainer"] > .main {{
 #background-image: url("data:image/png;base64,{img}");
 #background-size: 100%;
 #background-position: top right;
 #background-repeat: repeat;
 #background-attachment: local;
 #}}

 #[data-testid="stSidebar"] > div:first-child {{
 #background-image: url("https://wallpapers.com/images/hd/dark-blue-plain-thxhzarho60j4alk.jpg");
 #background-size: 150%;
 #background-position: top left; 
 #background-repeat: repeat;
 #background-attachment: fixed;
 #}}

 #[data-testid="stHeader"] {{
 #background: rgba(0,0,0,0);
 #}}

 #[data-testid="stToolbar"] {{
 #right: 2rem;
 #}}
 #</style>
 #"""
 #st.markdown(page_bg_img, unsafe_allow_html=True)

 #css="""
 #     <style>
 #           [data-testid="stForm"] {
 #              background: Purple;
 #           }
 #     </style>
 #     """


