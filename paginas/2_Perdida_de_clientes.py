import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import requests
from PIL import Image
import urllib.request

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

def show_page():

 @st.cache_data
 def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

 churn_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv")
 f1_knn=Image.open("f1_knn.png")
 error_knn=Image.open("error_knn.png")


#st.subheader("Clasificación de Posible Abandono de Clientes de Telecomunicaciones")
 st.title("Perdida de clientes")
 st.write(":green[*Modelo ML - Clasificación*] ")
 pagina1, pagina2, pagina3 =st.tabs(["Home","Predicción individual","Predicción múltiple"])

 with pagina1:

     st.subheader("Exploración y Análisis")

     st.write("La predicción de perdida de clientes se hace por medio de una modelo de machine learning que permite hacer clasificaciones, en este caso a los clientes se les clasifica según sus características que podrían definirlo como posible cliente que abandonara la empresa.")

     st.write("Los datos presentados son de una empresa de telecomunicaciones, los cuales contienen información relevante para saber si el cliente podría permanecer o abandonar la empresa, los datos son registros históricos de la empresa con las decisiones de los clientes. Mediante estos datos se puede saber si un cliente podría abandonar la empresa, ya que pueden cumplir perfiles parecidos. Los resultados son importantes, ya que permite a la empresa saber con anticipación cuales son los clientes que podrían abandonar la empresa, en base en esto se pueden crear estrategias para que esos clientes permanezcan el mayor tiempo posible en la empresa.")
     
     st.write("Los datos se presentan a continuación:")
     
     st.code("""
        churn_df.head(5)
        churn_df.shape     
     """)
     
     st.table(churn_df.head(5))
     st.write(churn_df.shape)

     st.write("Los datos contienen columnas información de datos personales de los clientes y de los servicios que contrato, en los datos personales se ve información tanto de sus ingresos, nivel de estudio, edad, etc. Mientras que en los de servicios muestran información si es que contrato ciertos servicios de internet, llamadas, además de cuanto dinero pago por esos servicios. Al final de la tabla se encuentra la información 'churn', que significa si es que ese cliente abandono la empresa, (0) no la abandono y (1) si la abandono.")
     
     st.write("Se eliminan las columnas que no se usaran que son 'loglong', 'logtoll', 'lninc' porque son información repetidas de  otras columnas, son simplemente los logaritmos de aquellas columnas.")
 
     st.code("""
       churn_df = churn_df.drop(["loglong", "lninc", "logtoll"], axis=1)
     """)
     
     st.write("En este caso los datos están limpiados y codificados por lo que no hay que hacer más limpieza.")
      
     st.subheader("Ingeniería de características")
     
     st.write("Lo más importante es ver como se comportan las características con respecto a la etiqueta en este caso 'churn' y algunas relaciones muy resumidas")
     
     st.write("Empezando por mostrar cual es la proporción de clientes que abandonan respecto a los que permanecen.")
     
     st.code("""
         churn_df["churn"] = churn_df["churn"].replace({1:"abandona", 0:"permanece"})
         grupo = pd.DataFrame(churn_df.groupby(["churn"], as_index=False)["income"].agg("count"))
     
         fig, ax =plt.subplots()
         ax.bar(grupo["churn"], grupo["income"], width=0.5, edgecolor="black", linewidth=0.3)
         ax.grid(alpha=0.2)
         ax.set_title("Perdida de clientes")
         ax.set_xlabel("Permanecia del cliente")
         ax.set_ylabel("Cantidad de clientes")
         plt.show()
     
     """)     
          
     
     churn_df["churn"] = churn_df["churn"].replace({1:"abandona", 0:"permanece"})
     agrupado = pd.DataFrame(churn_df.groupby(["churn"], as_index=False)["income"].agg("count"))
     
     fig, ax =plt.subplots()
     ax.bar(agrupado["churn"], agrupado["income"], width=0.5, edgecolor="black", linewidth=0.3)
     ax.grid(alpha=0.2)
     ax.set_title("Perdida de clientes")
     ax.set_xlabel("Permanecia del cliente")
     ax.set_ylabel("Cantidad de clientes")
     st.pyplot(fig)
     
     st.write("Esto muestra que la mayoría de los clientes permanecen con la empresa, la relación no siempre es igual en todas las empresas, esto provocara una mejor predicción para los clientes que se quedan en la empresa, puesto que hay más cantidad de perfiles para comparar versus a los que abandonan.")
     
     st.write("Por otro lado existe un valor anormal en los ingresos 'income' que casi cuadriplicaba a su antecesor, esto podría provocar que el modelo no haga buenas interpretaciones de los datos y por lo tanto no tan buenas predicciones. Por lo tanto este dato anormal es eliminado.")     
     
     st.code("""
         fig, ax =plt.subplots()
         ax.hist(churn_df["income"], bins=20, edgecolor="black", linewidth=0.3)
         ax.grid(alpha=0.2)
         ax.set_title("Distribución de ingresos de los clientes")
         ax.set_xlabel("Ingresos de los clientes ($)")
         ax.set_ylabel("Cantidad de clientes")
         plt.show()
     """)
     
     left, right = st.columns(2)
     
     with left:
        fig, ax =plt.subplots(figsize=(8,8.35))
        ax.hist(churn_df["income"], bins=20, edgecolor="black", linewidth=0.3)
        ax.grid(alpha=0.2)
        ax.set_title("Distribución de ingresos de los clientes")
        ax.set_xlabel("Ingresos de los clientes ($)")
        ax.set_ylabel("Cantidad de clientes")
        st.pyplot(fig)
        
     
     churn_df_mod=churn_df[churn_df["income"]<500]
     
     with right:
        fig, ax =plt.subplots(figsize=(8,8))
        ax.hist(churn_df_mod["income"], bins=25, edgecolor="black", linewidth=0.3)
        ax.grid(alpha=0.2)
        ax.set_title("Distribución de ingresos de los clientes")
        ax.set_xlabel("Ingresos de los clientes")
        ax.set_ylabel("Cantidad de clientes")
        st.pyplot(fig)
     
     churn_df["churn"] = churn_df["churn"].replace({"abandona":1, "permanece":0})
     churn_df_mod["churn"] = churn_df_mod["churn"].replace({"abandona":1, "permanece":0})
     
     st.write("Nos aseguramos que las columnas no tengan otro dato anormal, para que no afecte las predicciones.")
     
     st.code("""
        fig, ax =plt.subplots(figsize=(8,8))
        ax.boxplot(churn_df)
        ax.grid(alpha=0.2)
        ax.set_title("Distribución de ingresos de los clientes")
        ax.set_xlabel("Ingresos de los clientes")
        ax.set_ylabel("Cantidad de clientes")
        plt.show() 
     """)
     churn_df_mod = churn_df_mod.drop(["loglong", "lninc", "logtoll"], axis=1)  
     fig, ax =plt.subplots(figsize=(8,8))
     ax.boxplot(churn_df_mod)
     ax.grid(alpha=0.2)
     ax.set_title("Distribución de ingresos de los clientes")
     ax.set_xlabel("Ingresos de los clientes")
     ax.set_ylabel("Cantidad de clientes")
     st.pyplot(fig)    
     
     st.write("Existe otro dato que es anormal, se encuentra muy alejado de los demás por lo que es preferible eliminarlo para una mejor interpretación del modelo, el dato corresponde a la columna 'cardten', el dato es eliminado.")
     
     churn_df_mod2=churn_df_mod[churn_df_mod["cardten"]<6000]
     
     st.write("Nos queda una tabla con los siguientes valores")
     
     st.table(churn_df_mod2.head(5))
     st.write(churn_df_mod2.shape)
     
     st.write("Separamos en 'feature' como todos las columnas excepto 'churn' y 'label' como 'churn', definidos como 'X' e 'y' respectivamente, además hacemos la separación entre datos de entrenamiento y prueba.")
     
     st.code("""
     from sklearn.model_selection import train_test_split
     X = churn_df.drop(["churn"], axis=1)
     y = churn_df['churn'].astype('int')    
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
     """)
     from sklearn.model_selection import train_test_split
     X_mod = churn_df_mod2.drop(["churn"], axis=1)
     y_mod = churn_df_mod2['churn'].astype('int')    
     X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, test_size=0.2, random_state=1)
     
     st.subheader("-Selección de modelo")

     st.write("En esta sección se probarán todos los modelos para saber cual es el que da mejor resultado justo a la elección de mejores parámetros gracias a GridSearch.")
     
     st.write("Empezando por el rendimiento de :red[**Árbol de decisión**]:")

     st.code("""
     from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, roc_auc_score, f1_score
     from sklearn.tree import DecisionTreeClassifier
     params_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 15, 20],'min_samples_leaf': [1, 2, 5]}
     dt = DecisionTreeClassifier(random_state=123)
     grid_search = GridSearchCV(dt, param_grid = params_grid, scoring='accuracy', cv = 5, verbose = False)
     grid_search.fit(X_train, y_train)
     best_params = grid_search.best_params_
     print(best_params)
     #Mejores parametros {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1}
     dt2 = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=1,random_state=123)
     dt2.fit(X_train, y_train)
     y_pred_dt = dt2.predict(X_test)
     precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_dt, beta=5, pos_label=1, average='weighted')
     auc = roc_auc_score(y_test, y_pred_dt, average='weighted')
     accuracy = accuracy_score(y_test, y_pred_dt)
     print("Modelo Arbol de decision")
     print(f"Accuracy is: {accuracy:.2f}")
     print(f"Precision is: {precision:.2f}")
     print(f"Recall is: {recall:.2f}")
     print(f"Fscore is: {f_beta:.2f}")
     print(f"AUC is: {auc:.2f}")
     """)

     st.write("-Accuracy es: :green[0.65] -Recall es: :green[0.65] -AUC es: :green[0.63]") 
     st.write("-Precision es: :green[0.72] -Fscore es: :green[0.65]")

     st.write("Mostrando el rendimiento ahora de :red[**Regresión logística**], en este caso no se puede ocupar GridSearch, puesto que los parámetros que contiene son más bien diferentes formas de cálculo, entre parámetros no son todos son compatibles.")

     st.code("""
         from sklearn.linear_model import LogisticRegression
         logR2 = LogisticRegression(random_state=123, solver="liblinear", penalty="l2", C=0.00001)
         logR2.fit(X_train, y_train)
         y_pred_logR2 = logR2.predict(X_test)
         precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_logR2, beta=5, pos_label=1, average='weighted')
         auc = roc_auc_score(y_test, y_pred_logR2, average='weighted')
         accuracy = accuracy_score(y_test, y_pred_logR2)
         print(f"Accuracy is: {accuracy:.2f}")
         print(f"Precision is: {precision:.2f}")
         print(f"Recall is: {recall:.2f}")
         print(f"Fscore is: {f_beta:.2f}")
         print(f"AUC is: {auc:.2f}")
     """)
      
     st.write("-Accuracy es: :green[0.75] -Recall es: :green[0.75] -AUC es: :green[0.50]") 
     st.write("-Precision es: :green[0.56] -Fscore es: :green[0.74]")
     
     st.write("Mostrando ahora el rendimiento de :red[**KNN**] o :red[**K vecinos cercanos**], este caso es un poco diferentes, puesto que se podría elegir el mejor K según cierta puntuación, pero eso no siempre coincide con el que tiene menos error, por eso se debe ver de manera gráfica.")
     
     left, right = st.columns(2)
     
     with left:
         st.write(f1_knn)
     with right:
         st.write(error_knn)
     
     st.write("El gráfico de la izquierda muestra que el K que tiene la mejor puntuación hay 5 valores que tienen la puntuación máxima en f1, esos valores de K son 8, 9, 27,35,36. En la gráfica derecha vemos que los codos de error se producen en este caso en los mismos valores, pero no siempre es así por ello se debe verificar. Nos quedamos con el valor más bajo que es 8.")
     
     st.code("""
         knn = KNeighborsClassifier(n_neighbors=8, weights='distance')
         knn.fit(X_train, y_train)
         y_pred_knn = knn.predict(X_test)
         precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_knn, beta=5, pos_label=1,  average='weighted')
         auc = roc_auc_score(y_test, y_pred_knn, average='weighted')
         accuracy = accuracy_score(y_test, y_pred_knn)
         print(f"Accuracy is: {accuracy:.2f}")
         print(f"Precision is: {precision:.2f}")
         print(f"Recall is: {recall:.2f}")
         print(f"Fscore is: {f_beta:.2f}")
         print(f"AUC is: {auc:.2f}")
     """)
     
     st.write("-Accuracy es: :green[0.62] -Recall es: :green[0.62] -AUC es: :green[0.58]") 
     st.write("-Precision es: :green[0.68] -Fscore es: :green[0.63]")
     
     st.write("Seguimos con :red[SVC] para saber que rendimiento con los datos, en este caso si se puede ocupar GridSearch.")
     
     st.code("""
         from sklearn.svm import SVC
         params_grid = {'C': [0.00001, 0.001,0.01,1, 10, 100], 'kernel': ['poly', 'rbf', 'sigmoid']}
         svc = SVC()
         grid_search = GridSearchCV(estimator = svc, param_grid = params_grid, scoring='f1', cv = 5, verbose = 1)
         grid_search.fit(X_train, y_train)
         best_params = grid_search.best_params_
         print(best_params)
         #Mostrando como mejor parámetros C=10 y kernel='rbf'
         svc2= SVC(kernel="rbf", C=10)
         svc2.fit(X_train, y_train)
         y_pred_svc = svc2.predict(X_test)
         precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_svc, beta=5, pos_label=1, average='weighted')
         auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_svc,  classes=[1,2,3]), average='weighted')
         accuracy = accuracy_score(y_test, y_pred_svc)
         print(f"Accuracy is: {accuracy:.2f}")
         print(f"Precision is: {precision:.2f}")
         print(f"Recall is: {recall:.2f}")
         print(f"Fscore is: {f_beta:.2f}")
         print(f"AUC is: {auc:.2f}")
     """)
     
     st.write("-Accuracy es: :green[0.55] -Recall es: :green[0.55] -AUC es: :green[0.50]") 
     st.write("-Precision es: :green[0.62] -Fscore es: :green[0.55]")
     
     st.write("Continuando con :red[**Random Forest**]:")
     
     st.code("""
         from sklearn.ensemble import RandomForestClassifier
         param_grid = {'n_estimators': [2*n+1 for n in range(20)], 'max_depth' : [2*n+1 for n in range(10) ], 'max_features':["auto", "sqrt", "log2"]}
         RFC = RandomForestClassifier()
         search = GridSearchCV(estimator=RFC, param_grid=param_grid,scoring='accuracy', cv=5)
         search.fit(X_train, y_train)
         print(search.best_params_)
         #Como resultado da: {'max_depth': 9, 'max_features': 'sqrt', 'n_estimators': 11}
         RFC2 = RandomForestClassifier(max_depth=9, max_features="sqrt" , n_estimators=11, random_state=123)
         RFC2.fit(X_train,y_train)
         y_pred_RFC2 = RFC2.predict(X_test)
         precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_RFC2, beta=5, pos_label=1, average='weighted')
         auc = roc_auc_score(y_test, y_pred_RFC2, average='weighted')
         accuracy = accuracy_score(y_test, y_pred_RFC2)
         print(f"Accuracy is: {accuracy:.2f}")
         print(f"Precision is: {precision:.2f}")
         print(f"Recall is: {recall:.2f}")
         print(f"Fscore is: {f_beta:.2f}")
         print(f"AUC is: {auc:.2f}") 
     """)
     st.write("-Accuracy es: :green[0.78] -Recall es: :green[0.78] -AUC es: :green[0.65]") 
     st.write("-Precision es: :green[0.76] -Fscore es: :green[0.77]")
     
     st.write("Ahora seleccionando el modelo :red[**ExtraTrees**] que en este caso debemos ver los errores que tienen las diferentes cantidades de árboles primero, el que tenga menos error, será el mejor.")
     
     st.code("""
        from sklearn.ensemble import ExtraTreesClassifier
        EF = ExtraTreesClassifier(oob_score=True, random_state=20, warm_start=True, bootstrap=True, n_jobs=-1)
        oob_list = list()
        for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
                   EF.set_params(n_estimators=n_trees)
                   EF.fit(X_train, y_train)
        #oob error
        oob_error = 1 - EF.oob_score_
        oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))
        et_oob_df = pd.concat(oob_list, axis=1).T.set_index('n_trees')
        #Muestra los errores para extra tree
        print(et_oob_df)
        #La cantidad de arboles que obtien menos error es 100 y 300, pero 100 por ser menor será mas óptimo
        ETC2 = ExtraTreesClassifier(oob_score=True, random_state=20, warm_start=True, bootstrap=True, n_jobs=-1, n_estimators=100)
        ETC2.fit(X_train, y_train)
        y_pred_ETC2 = ETC2.predict(X_test)
        precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ETC2, beta=5, pos_label=1, average='weighted')
        auc = roc_auc_score(y_test, y_pred_ETC2, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_ETC2)
        print(f"Accuracy is: {accuracy:.2f}")
        print(f"Precision is: {precision:.2f}")
        print(f"Recall is: {recall:.2f}")
        print(f"Fscore is: {f_beta:.2f}")
        print(f"AUC is: {auc:.2f}")
     """)
     
     st.write("-Accuracy es: :green[0.82] -Recall es: :green[0.82] -AUC es: :green[0.72]") 
     st.write("-Precision es: :green[0.81] -Fscore es: :green[0.82]")
     
     st.write("Ahora ocupando :red[**GradientBoosting**]:")
     
     st.code("""
        from sklearn.ensemble import GradientBoostingClassifier
        #El menor error son 100 y menos arboles
        error_list = list()
        tree_list = [1,5,10,50,100,150,200,400]
        for n_trees in tree_list:
            GBC = GradientBoostingClassifier(n_estimators=n_trees, random_state=42)
            GBC.fit(X_train, y_train)
            y_pred = GBC.predict(X_test)
            error = 1.0 - accuracy_score(y_test, y_pred)
            error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))
        error_df = pd.concat(error_list, axis=1).T.set_index('n_trees')
        print(error_df)
        param_grid = {'n_estimators': tree_list, 'learning_rate': [0.1, 0.01, 0.001, 0.0001], 'subsample': [0.5], 'max_features': [4]}
        GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=param_grid, scoring='accuracy', n_jobs=-1)
        GV_GBC = GV_GBC.fit(X_train, y_train)
        print(GV_GBC.best_estimator_)
        #Se obtuvieron los errores de los diferentes arboles, los mejores parametros son {max_features=4, n_estimators=10, random_state=42, subsample=0.5}
        GBC1 = GradientBoostingClassifier(n_estimators=10, max_features=4, learning_rate=0.001, random_state=42, subsample=0.5)
        GBC1.fit(X_train, y_train)
        y_pred_GBC1 = GBC1.predict(X_test)
        precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_GBC1, beta=5, pos_label=1, average='weighted')
        auc = roc_auc_score(y_test, y_pred_GBC1, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_GBC1)
        print(f"Accuracy is: {accuracy:.2f}")
        print(f"Precision is: {precision:.2f}")
        print(f"Recall is: {recall:.2f}")
        print(f"Fscore is: {f_beta:.2f}")
        print(f"AUC is: {auc:.2f}")
     """)
     
     st.write("-Accuracy es: :green[0.75] -Recall es: :green[0.75] -AUC es: :green[0.50]") 
     st.write("-Precision es: :green[0.56] -Fscore es: :green[0.74]")
     
     st.write("Ahora con el optimizador :red[**AdaBoostClassifier**] en este caso ocupando con el mejor modelo hasta el momento ExtraTrees:")
     
     st.code("""
        from sklearn.ensemble import AdaBoostClassifier
        ABC = AdaBoostClassifier(ExtraTreesClassifier())
        param_grid = {'n_estimators': [1,2,5,10,50,100,200], 'learning_rate': [0.0001,0.001,0.01,0.1, 1]}
        GV_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring='accuracy', n_jobs=-1)
        GV_ABC = GV_ABC.fit(X_train, y_train)
        print(GV_ABC.best_estimator_) 
        #Obteniendo que el optimizador debe tener:  learning_rate=0.001, n_estimators=200
        ABC1 = AdaBoostClassifier(ExtraTreesClassifier(oob_score=True, random_state=20, warm_start=True, bootstrap=True,n_jobs=-1,n_estimators=100), n_estimators=200, learning_rate=0.001)
        ABC1.fit(X_train, y_train)
        y_pred_ABC1 = ABC1.predict(X_test)
        precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ABC1, beta=5, pos_label=1, average='weighted')
        auc = roc_auc_score(y_test, y_pred_ABC1, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_ABC1)
        print(f"Accuracy is: {accuracy:.2f}")
        print(f"Precision is: {precision:.2f}")
        print(f"Recall is: {recall:.2f}")
        print(f"Fscore is: {f_beta:.2f}")
        print(f"AUC is: {auc:.2f}")   
     """)
     st.write("-Accuracy es: :green[0.75] -Recall es: :green[0.75] -AUC es: :green[0.67]") 
     st.write("-Precision es: :green[0.75] -Fscore es: :green[0.75]")
     
     st.write("Ahora con :red[**BaggingClassifier**] con el estimador árbol de decisión:")
     
     st.code("""
        from sklearn.ensemble import BaggingClassifier
        param_grid = {'n_estimators': [2*n+1 for n in range(20)], 'estimator__max_depth' : [2*n+1 for n in range(10) ] }
        Bag = BaggingClassifier(DecisionTreeClassifier(), random_state=20, bootstrap=True)
        search = GridSearchCV(estimator=Bag, param_grid=param_grid, scoring='accuracy', cv=3)
        search.fit(X_train, y_train)
        print(search.best_params_)
        #Encontro que los mejores parametros son n_estimators=35 y max_depth=7
        Bag = BaggingClassifier(DecisionTreeClassifier(criterion="gini", max_depth=7, random_state=123), n_estimators=35, random_state=0, bootstrap=True)
        Bag.fit(X_train, y_train)
        y_pred_bag_dt = Bag.predict(X_test)
        precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_bag_dt, beta=5, pos_label=1, average='weighted')
        auc = roc_auc_score(y_test, y_pred_bag_dt, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_bag_dt)
        print(f"Accuracy is: {accuracy:.2f}")
        print(f"Precision is: {precision:.2f}")
        print(f"Recall is: {recall:.2f}")
        print(f"Fscore is: {f_beta:.2f}")
        print(f"AUC is: {auc:.2f}")     
     """)
     st.write("-Accuracy es: :green[0.62] -Recall es: :green[0.62] -AUC es: :green[0.62]") 
     st.write("-Precision es: :green[0.71] -Fscore es: :green[0.63]")
     
     st.write("Mostrando ahora :red[**StackingClassifier**], en este caso se mezclarán tres estimadores ExtraTrees, RandomForest, DecisionTree con las mismas configuraciones anteriores encontradas por GridSearch.")
     
     st.code("""
        from sklearn.ensemble import StackingClassifier
        estimators = [('ETC',ExtraTreesClassifier(oob_score=True, random_state=20, warm_start=True, bootstrap=True,n_jobs=-1,n_estimators=100)),('RF',RandomForestClassifier(max_depth=7, max_features="sqrt" , n_estimators=9, random_state=20)), ('dt',DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=1,random_state=20))]
        clf = StackingClassifier( estimators=estimators, final_estimator= LogisticRegression())
        clf.fit(X_train, y_train)
        y_pred_clf = clf.predict(X_test)
        precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_clf, beta=5, pos_label=1, average='weighted')
        auc = roc_auc_score(y_test, y_pred_clf, average='weighted')
        accuracy = accuracy_score(y_test, y_pred_clf)
        print(f"Accuracy is: {accuracy:.2f}")
        print(f"Precision is: {precision:.2f}")
        print(f"Recall is: {recall:.2f}")
        print(f"Fscore is: {f_beta:.2f}")
        print(f"AUC is: {auc:.2f}")
     """)
     
     st.write("-Accuracy es: :green[0.82] -Recall es: :green[0.82] -AUC es: :green[0.72]") 
     st.write("-Precision es: :green[0.81] -Fscore es: :green[0.82] ")
     
     st.subheader("Conclusión")
     
     st.write("En conclusión es posible que los datos no sean la cantidad suficiente para sacar puntuaciones altas, en general las puntuaciones fueron bajas, excepto dos modelos que lograron sacar puntuaciones aceptables, como es :red[ExtraTrees] y :red[Stacking], que sacaron puntuaciones idénticas, por ello la selección final es elegir :red[ExtraTrees] solo por simplicidad.")
     
 with pagina2:

     st.write(":blue[Rellene el formulario y haga su predicción]")     

     st.sidebar.write(":blue[Rellene el formulario y descubra si un cliente puede abandonar la empresa]")

     st.header("Formulario con los datos del cliente:")

     X = churn_df.drop(["churn", "loglong", "lninc", "logtoll"], axis=1)
     y = churn_df['churn'].astype('int')

     with st.form("cliente", clear_on_submit=False, border=True):
            churn_df["callcard"] = churn_df["callcard"].replace({1:"Si", 0:"No"})
            churn_df[["equip"]] = churn_df[["equip"]].replace({1:"Si", 0:"No"})
            churn_df["wireless"] = churn_df["wireless"].replace({1:"Si", 0:"No"})
            churn_df["voice"] = churn_df["voice"].replace({1:"Si", 0:"No"})
            churn_df["pager"] = churn_df["pager"].replace({1:"Si", 0:"No"})
            churn_df["internet"] = churn_df["internet"].replace({1:"Si", 0:"No"})
            churn_df["callwait"] = churn_df["callwait"].replace({1:"Si", 0:"No"})
            churn_df["confer"] = churn_df["confer"].replace({1:"Si", 0:"No"})
            churn_df["ebill"] = churn_df["ebill"].replace({1:"Si", 0:"No"})
            churn_df["ed"] = churn_df["ed"].replace({1:"Educación Básica", 2:"Educación Media", 3:"Educación Superior", 4:"Magistrado/a", 5:"Doctorado/a"})
            st.subheader("Datos personales del cliente:")
            Nombre=st.text_input(label="Escriba el nombre del cliente")           
            #marca_auto=st.selectbox("Marca del vehículo", data3["Brand"].unique())   
                     
            left_column, right_column=st.columns(2)
            with left_column:
                          income=st.select_slider("Ingreso anual del cliente", np.sort(churn_df["income"].unique()))
                          numero_años_en_residencia=st.select_slider("Años en su vivienda actual", churn_df["address"].unique()) 
                          
                          edad_del_cliente=st.select_slider("Edad del cliente", np.sort(churn_df["age"].unique()))
                                             
            with right_column:
                          nivel_de_estudio=st.selectbox("Nivel de estudio", churn_df["ed"].unique(), index=True)                               
                          
                          tenure=st.select_slider("Meses que el cliente ha permanecido en la empresa del servicio", np.sort(churn_df["tenure"].unique()))
                          años_trabajados=st.select_slider("Años trabajando", np.sort(churn_df["employ"].unique()))
            st.subheader("Datos del servicio del cliente:")
            left_column, right_column, three_column=st.columns(3)
            with left_column:
                          servicio_de_internet=st.selectbox("¿Contrato servicios de internet?", np.sort(churn_df["internet"].unique()))
                          
                          equipo_de_empresa=st.selectbox("¿Tiene equipos de la empresa?", np.sort(churn_df["equip"].unique()))
                          
                          
                          
                          callwait=st.selectbox("¿Contrato servicio de llamada en espera?", np.sort(churn_df["callwait"].unique()))
                          
                          longmon=st.select_slider("Gastos mensuales en llamadas larga distancia", np.sort(churn_df["longmon"].unique()))
                          cardmon=st.select_slider("Gastos mensuales en servicios de llamadas", np.sort(churn_df["cardmon"].unique()))
                          tollten=st.select_slider("Gastos totales de peajes", np.sort(churn_df["tollten"]))
                          
            with right_column:
                          servicio_inalambricos=st.selectbox("¿Contrato servicios inalámbricos?", np.sort(churn_df["wireless"].unique()))
                          
                          servicio_de_voz=st.selectbox("¿Contrato servicio de correo de voz?", np.sort(churn_df["voice"].unique()))
                          
                          confer=st.selectbox("¿Contrato el servicio de conferencias?", np.sort(churn_df["confer"].unique()))
                          
                          tollmon=st.select_slider("Gastos mensuales en peajes (Toll charger)", np.sort(churn_df["tollmon"].unique()))
                          wiremon=st.select_slider("Gastos mensuales en servicios inalámbricos", np.sort(churn_df["wiremon"].unique()))
                          cardten=st.select_slider("Gastos totales en llamadas", np.sort(churn_df["cardten"].unique()))
            with three_column:
                          servicio_telefonico=st.selectbox("¿Contrato servicios telefónicos?", np.sort(churn_df["callcard"].unique()))
                          
                          servicio_localizador=st.selectbox("¿Tiene un localizador?", np.sort(churn_df["pager"].unique()))
                          
                          ebill=st.selectbox("¿El cliente utiliza facturación electrónica?", np.sort(churn_df["ebill"].unique()))
                          
                          equipmon=st.select_slider("Gastos mensuales en alquiler de equipos", np.sort(churn_df["equipmon"].unique()))
                          longten=st.select_slider("Gastos totales de llamadas larga distancia", np.sort(churn_df["longten"].unique()))
                          custcat=st.selectbox("Plan contratado", np.sort(churn_df["custcat"].unique()))
            variable_input_sumit = st.form_submit_button("Enviar")
                          




#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless", "longmon", "tollmon", "equipmon", "cardmon", "wiremon", "longten", "tollten", "cardten", "voice", "pager", "internet", "callwait", "confer", "ebill", "custcat"]]



#callcard, wireless, employ son otras clasificaciones por ende no sirven, los otros son solo numeros
#X=churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', "callcard", "wireless"]]


#y = churn_df['churn']
#print(y.head())


     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
     #from sklearn.ensemble import AdaBoostClassifier     
     from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, f1_score
     #from sklearn.preprocessing import label_binarize
 
     from sklearn.tree import DecisionTreeClassifier


     from sklearn.ensemble import ExtraTreesClassifier


     ETC2 = ExtraTreesClassifier(oob_score=True, 
                          random_state=5, 
                          warm_start=True,
                          bootstrap=True,
                          n_jobs=-1,
                          n_estimators=30)
     ETC2.fit(X_train, y_train)

     y_pred_ETC2 = ETC2.predict(X_test)

     precision, recall, f_beta, support = precision_recall_fscore_support(y_test, y_pred_ETC2, beta=5, pos_label=1, average='weighted')
     auc = roc_auc_score(y_test, y_pred_ETC2, average='weighted')
#auc = roc_auc_score(label_binarize(y_test, classes=[1,2,3]), label_binarize(y_pred_ETC2,  classes=[1,2,3]), average='weighted')
     accuracy = accuracy_score(y_test, y_pred_ETC2)
#st.write("Extra Tree")
#st.write(f"Accuracy is: {accuracy:.2f}")
#st.write(f"Precision is: {precision:.2f}")
#st.write(f"Recall is: {recall:.2f}")
#st.write(f"Fscore is: {f_beta:.2f}")
#st.write(f"AUC is: {auc:.2f}")

     if variable_input_sumit:
        data_3=pd.DataFrame([Nombre, tenure, edad_del_cliente, numero_años_en_residencia, income, nivel_de_estudio, años_trabajados , equipo_de_empresa, servicio_telefonico, servicio_inalambricos, longmon, tollmon, equipmon, cardmon, wiremon, longten, tollten, cardten, servicio_de_voz, servicio_localizador, servicio_de_internet, callwait, confer, ebill, custcat]).T
        data_3=data_3.rename(columns={0:"Nombre del cliente", 1:"tiempo en la empresa", 2:"Edad", 3:"años en residencia", 4:"ingreso anual", 5:"nivel educativo", 6:"Años de trabajo", 7:"Posee equipos de la empresa", 8:"Posee servicios de llamadas", 9:"Posee servicios inalambricos", 10:"Gastos mensuales en llamadas larga distancias", 11:"Gastos mensuales de peaje", 12:"Gastos mensuales en equipos", 13:"Gastos mensules en llamadas", 14:"Gastos mensuales en servicios inalambricos", 15:"Gastos totales en llamadas larga distancias", 16:"Gastos totales en peajes", 17:"Gastos totales en llamadas", 18:"Servicio de buzon de voz", 19:"Servicio de localizador", 20:"Servicio de internet", 21:"Serivcio de llamadas en espera", 22:"Servicio de conferencias", 23:"Utiliza facturación electronica", 24:"Plan del cliente"})
        
        st.write(f":blue[Los datos del cliente seleccionados son:]")                       
        st.write(data_3)
        
        if  nivel_de_estudio=="Educación Básica":
                     nivel_de_estudio=1
        if  nivel_de_estudio=="Educación Media":
                     nivel_de_estudio=2  
        if  nivel_de_estudio=="Educación Superior":
                     nivel_de_estudio=3
        if  nivel_de_estudio=="Magistrado/a":
                     nivel_de_estudio=4
        if  nivel_de_estudio=="Doctorado/a":
                     nivel_de_estudio=5 
                     
        if servicio_de_internet=="Si":
                     servicio_de_internet=1
        if servicio_de_internet=="No":
                     servicio_de_internet=0
        
        if equipo_de_empresa=="Si":
                     equipo_de_empresa=1
        if equipo_de_empresa=="No":
                     equipo_de_empresa=0
        
        if callwait=="Si":
                     callwait=1
        if callwait=="No":
                     callwait=0
        
        if servicio_inalambricos=="Si":
                     servicio_inalambricos=1
        if servicio_inalambricos=="No":
                     servicio_inalambricos=0
        
        if servicio_de_voz=="Si":
                     servicio_de_voz=1
        if servicio_de_voz=="No":
                     servicio_de_voz=0
        
        if confer=="Si":
                     confer=1
        if confer=="No":
                     confer=0
        
        if servicio_telefonico=="Si":
                     servicio_telefonico=1
        if servicio_telefonico=="No":
                     servicio_telefonico=0
        
        if servicio_localizador=="Si":
                     servicio_localizador=1
        if servicio_localizador=="No":
                     servicio_localizador=0
        
        if ebill=="Si":
                     ebill=1
        if ebill=="No":
                     ebill=0
                        
        data_2=pd.DataFrame([tenure, edad_del_cliente, numero_años_en_residencia, income, nivel_de_estudio, años_trabajados , equipo_de_empresa, servicio_telefonico, servicio_inalambricos, longmon, tollmon, equipmon, cardmon, wiremon, longten, tollten, cardten, servicio_de_voz, servicio_localizador, servicio_de_internet, callwait, confer, ebill, custcat]).T
        data_2=data_2.rename(columns={0:"tenure", 1:"age", 2:"address", 3:"income", 4:"ed", 5:"employ", 6:"equip", 7:"callcard", 8:"wireless", 9:"longmon", 10:"tollmon", 11:"equipmon", 12:"cardmon", 13:"wiremon", 14:"longten", 15:"tollten", 16:"cardten", 17:"voice", 18:"pager", 19:"internet", 20:"callwait", 21:"confer", 22:"ebill", 23:"custcat"}) 
        
        y_pred_ETC3 = ETC2.predict(data_2)
        
        if y_pred_ETC3==0:
              y_pred_ETC3=f":green[permanezca] en la empresa"
              mensaje=":blue[Felicitaciones estás entregando un buen servicio]"
        
        if y_pred_ETC3==1:
              y_pred_ETC3=":red[abandoné] la empresa"      
              mensaje=":blue[Recomendamos crear estrategias para mantener la permanencia del cliente]"    
        
        st.header(f"Es probable que el cliente {Nombre} {y_pred_ETC3}")
        
        st.write(mensaje)
        
        st.write(f"Modelo ML: :green[ExtraTree]") 
        st.write(f"Accuracy: :green[{round(accuracy,2)}]     Recall: :green[{round(recall,2)}]")
        st.write(f"Presición: :green[{round(precision,2)}]     Fscore: :green[{round(f_beta,2)}]")
               
        
 with pagina3:

     st.subheader("Instrucciones")
     
     st.write("-La tabla con los datos debe estar limpiada y debe contener solo datos numéricos")
     
     st.write("-Debe ser un archivo '.csv'")
     
     st.write("-La tabla debe contener todas estas columnas en el siguiente orden:")
     
     if st.checkbox("Mostrar columnas"):
     
                ''':green["tenure", "age", "address", "income", "ed", "employ", "equip", "callcard", "wireless", "longmon", "tollmon", "equipmon", "cardmon", "wiremon", "longten", "tollten", "cardten", "voice", "pager", "internet", "callwait", "confer", "ebill","custcat"]'''
     
     st.write("-Puede contener algunas columnas adicionales como 'loglong', 'lninc', 'logtoll', 'Name', 'churn'")
     
     st.write("-Puede descargar datos de prueba dando click [aquí](%s)" % "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv" )
     
     with st.form("carga archivo", clear_on_submit=True):
     
                data_up = st.file_uploader("Inserte un archivo", type=["csv"])
                
                recibido=st.form_submit_button("Enviar")
                
                if recibido and data_up is not None:
                      #ret = self.upload_file(data_up)
                      
                      #if ret is not False:
                      
                              data_up = pd.read_csv(data_up)
                      
                              data_up = pd.DataFrame(data_up)
                              
                              data_up = data_up.drop(["Name"], axis=1, errors="ignore")
                              
                              data_up = data_up.drop(["loglong"], axis=1, errors="ignore")
                              
                              data_up = data_up.drop(["lninc"], axis=1, errors="ignore")
                              
                              data_up = data_up.drop(["logtoll"], axis=1, errors="ignore")
                              
                              data_up = data_up.drop(["churn"], axis=1, errors="ignore")                     
     
                              y_pred_up = ETC2.predict(data_up)
                              
                              y_pred_up2 = pd.DataFrame(y_pred_up)
                              
                              mezcla = pd.DataFrame(pd.concat([data_up, y_pred_up2], axis=1))
                              mezcla = mezcla.rename(columns={0:"churn"})
                              
                              #st.write(mezcla)
                              
                                                                                        
                              
                              abandonan = []
                              permanecen = []     
     
                              for i in y_pred_up:
                                    if i ==1:
                                          abandonan.append(i)
                                    if i ==0:
                                          permanecen.append(i)
                              
                              #st.subheader("Pronostico:")
                              st.write("") 
                              st.subheader(f"- :red[{len(abandonan)}]/{len(abandonan)+len(permanecen)} clientes pueden ser perdidos")
                              #st.subheader(f"- :green[{len(permanecen)}]/{len(abandonan)+len(permanecen)} clientes pueden quedarse")
                             
                              
                              st.write(":blue[Los clientes que se pueden perder son: ]")
                              
                              st.write(mezcla[mezcla["churn"]==1])  
                  
                              st.write(":blue[(click arriba para descargar)]")
              
              
     
             









