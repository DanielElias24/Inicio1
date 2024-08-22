import base64
import streamlit as st
from PIL import Image

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

 st.title("Reconocimiento de videos")

 st.write(":green[*Modelo ML - YOLO - Tracking*]")

 st.subheader("Exploración y Análisis")

 st.write("El modelo YOLO es un modelo de tracking que se ocupa para la detección de objetos estáticos o en movimiento, el modelo ya esta pre entrenado, fue entrenado principalmente para reconocer personas u objetos grandes como vehículos, etc. Para reconocer otro tipo de objeto el modelo no los reconoce, pero es posible hacer un entrenamiento para que pueda reconocerlos.")

 st.write("El desafío de este modelo en si es que pueda reconocer cualquier tipo de cosas, con el desafío que puede ser ocupar archivos de video.")

 st.subheader("Datos de entrenamiento y video")

 st.write("El modelo se puede ocupar para una cámara en vivo con el siguiente código: ")

 st.code("""
    from ultralytics import YOLO
    model= YOLO("yolov8s.pt")
    results = model.predict(source="0", show=True)
 """)

 st.write("Simplemente con ese código se puede ocupar, con 'yolov8s.pt' siendo un modelo pre entrenado.")

 st.write("En caso de que se quiera entrenar y además ser ocupado para videos se requiere un poco más de elaboración porque el video requiere ser procesado.")

 st.write("En este caso se ocupará el siguiente video que ya ha sido visto por el modelo 'yolov8x.pt', pero no registra absolutamente nada, el video es el siguiente: ")

 st.video("girasoles.mp4")

 st.write("Para entrenar el modelo se ocupará los mismos datos de girasoles que se ocupo en reconocimiento de imagenes")

 st.subheader("Entrenamiento del modelo")

 st.write("Para entrenar el modelo se requieren datos compilados de una forma especial, estos datos compilados son obtenidos de una pagina web que es: ")

 st.write("https://www.cvat.ai/")

 st.write("En la pagina web se debe crear una cuenta, al crear una cuenta se mostrara un panel y se debe poner 'Crear dataset', nos preguntará nombre del dataset y los labels que queremos ocupar para la recolección de objetos, además mostrará un espacio para cargar todo los archivos que queremos ocupar como entrenamiento, en este caso se ocupara imagenes de girasoles.")

 st.write("El siguiente paso es muy importante porque no es solo cargar las imagenes de entrenamiento, sino que hay que señalar objeto por objeto que queremos que reconosca, ocupando diferentes labels imagen por imagen de nuestro entrenamiento. Esto hara que el modelo se ajuste a nuestras necesidades y tenga un mejor resultado puesto que podemos decidir los objetos que usará.")

 ejemplo=Image.open("ejemplo1.png")

 st.write("Ajustando imagenes de entrenamiento:")

 st.write(ejemplo)

 st.write("Al ajustar todas las imagenes se debe guardar los ajustes en la página, luego seleccionar el dataset creado en la parte de acciones, en ese lugar se debe seleccionar 'exportar', luego colocar el nombre del nombre del dataset y con el formato 'YOLO 0.1' que es el ultimo en la lista.")

 st.write("Adicionalmente se debe crear un archivo llamado 'config.yaml', este archivo debe estar configurado de la siguiente forma: ")

 st.code("""
   path: /home/daniel/Documentos/Proyectos data science/flores/
   train: images
   val: images
   names:
      0: Girasol
 """)

 st.write("Donde path es la ubicación de los datos de entrenamiento, donde debe estar una carpeta con las imagenes y otra carpeta dentro con los labels descargados, train y val es una segunda ubicación dentro de esas carpetas, esas ubicaciones que pueden ubicar a las mismas imagenes para entrenar 'trian' y validar 'val', names muestra las etiquetas que mostrará las imagenes, el valor de las etiquetas de cada imagen esta dentro del archivo de labels, es el primer valor.")

 st.write("Ahora bien que esta todo los datos listos, se debe crear un codigo donde el video se descomponga en frames y luego se codifique nuevvamente, cada frame sera una imagen que se analizará.")

 st.code("""
   video_path_out = "{}_out.mp4".format("girasoles.mp4")
   cap = cv2.VideoCapture("girasoles.mp4")
   ret, frame = cap.read()
   H, W, _ = frame.shape
   out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*"MP4V"), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
 """)

 st.write("Esto tiene que ver con la decomposición del video")

 st.write("En este nuevo codigo se aplican rectagulos de reconocimiento de objetos, tambien se entrena el modelo con las configuraciones deseadas, luego el video se codifica.")

 st.code("""
    model = YOLO("yolov8s.pt", "v8")
    model.train(data="config.yaml", epochs=3)
    threshold = 0.3
    while ret:
       results = model(frame)[0]
       for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_color = (0, 0, 0)
            text = f"{results.names[int(class_id)].upper()}"
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            if score > threshold:
                 cv2.rectangle(frame, (int(x1), int(y1 - text_height - 10)), (int(x1) + text_width, int(y1)), (0, 255, 255), cv2.FILLED)
                 cv2.putText(frame, text, (int(x1), int(y1 - 5)), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
       out.write(frame)
       ret, frame = cap.read()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
 """)

 st.write("Algo importante es threshold que es el umbral de calidad del objeto si es que supera ese umbral reconocera el objeto, por lo tanto entre mas alto, menos objetos reconocera y si el umbral es mas bajo, mas objeto reconocera, esto debe ser ocupado con discreción puesto que entre mas bajo más, probabilidad de equivocarse en el reconociemiento de objetos.")

 st.write("Al realizar el entrenamiento y aplicarlo al video se obtiene el siguiente resuldado:")

 st.video("girasoles2.webm") 

 st.write("El modelo hizo un buen trabajo logrando identificar la mayoria de los girasoles con mucha precisión. En cuanto a los girasoles que estan atras se necesitan mucho mas datos de entrenamiento para lograr identificarlas. Aparcen algunos errores muy pocas veces interpreta por muy poco tiempo que hay objetos cuando enrealidad no la hay, esto es producto que algunas imagenes de entrenamiento estaban de reversa, la flor por atras tiene un gran tallo verde, esto confunde al modelo identificando algunos tallos como girasoles por muy poco tiempo y alguna nubes por error.")

 st.subheader("Conclusión")

 st.write("El modelo hizo un gran trabajo identificando objetos, a pesar de que fue entrenado con pocos datos, los pequeño movimientos repentinos de los girasoles no pueden impedir que el modelo recnocesca los girasoles en todo momento. El modelo es capaz de identificar patrones difificiles sin importar el tamaño que posean.")






