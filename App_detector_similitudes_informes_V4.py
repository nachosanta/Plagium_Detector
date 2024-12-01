
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import re # regular expressions
import pandas as pd
#import PyPDF2
#from PyPDF2 import PdfReader # lector de pdfs
import pdftotext
#import unidecode
from unidecode import unidecode
#import numpy as np
from dash.exceptions import PreventUpdate
import base64
import io
import zipfile
#nltk.download('snowball_data') # ya lo baje
#import plotly
#from plotly.express import data
from nltk.tokenize import word_tokenize
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import json
#import sklearn
import os
import spacy # PARA LEMATIZAR EN ESPAÑOL Y DESPUES PARA VECTORIZAR CON GLOVE
nlp_spacy = spacy.load("es_core_news_md")  # Carga el modelo en español # modelo pre entrenado para lematizacion y vectorizacion GLove
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
# cargo un modelo de Doc2Vec
modelo_doc2vec = Doc2Vec.load("C:\\Users\HP\Desktop\Tesis\data_preprocesada_entrenamiento_doc2vec\modelo_doc2vec_informes_vectors50_window2_minc2_dm0_epochs40.model")
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import hierarchy
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
with open('C:\\Users\HP\Desktop\Tesis\gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

############################### LAYOUT

app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

# Inicializar eL lemmatizer de NLTK


# defino stopwords
spanish_stopwords = stopwords.words('spanish')


# armo los tabs cada uno con su funcionalidad
tab1_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Ranking pares de informes con alta similitud", className="card-text"),
            html.Hr(style={'border': '1.5px dashed #000000'}),
            html.Div(id='output-tabla-ranking'),
        ]
    ),
    className="mt-3",
)
tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("Grupos de 3 o más informes para revisar en conjunto", className="card-text"),
            html.Hr(style={'border': '1.5px dashed #000000'}),
            html.Div(id='output-tabla-grupos')
        ]
    ),
    className="mt-3",
)

# armo el layout de la app
app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dbc.Stack([
                    dbc.Stack([
                    html.Img(src='/assets/buscar-archivo.png', height=80, width=80),#, style={'background-color': 'white'}),
                    ],gap=1
                        ),
                    html.H1("Detector de Similitudes entre Informes", style={'fontSize': 50})],
                    direction="horizontal",
                    gap=3
                ),
                    width={"size":"auto", "order":1, "offset": 0}),
            html.Br(),
            ], justify='start'),
        dbc.Row([
            dbc.Col(html.H1("(logo y nombre de la institución cliente)", style={'fontSize': 13}),
                    width={"size":'auto', "order":2, "offset": 0})
            ], justify='start'),
        html.Hr(),
        dcc.Upload(
            id='upload-informes',
            children=html.Div([
                'Subir archivo ZIP con entregas (descargado del moodle):    ',
                html.A('      arrastrar o seleccionar aquí', id='aviso-carga-zip', style={'color': '#3FF43F', 'fontSize': 16})
            ], style={'fontSize': 18}),
            multiple=False
        ),
        html.A('', id='aviso-error', style={'color': 'red', 'fontSize': 16}),
        html.Br(),
        html.Br(),
        dcc.Upload(
            id='upload-enunciado',
            children=html.Div([
                'Subir archivo PDF con el enunciado (opcional):    ',
                html.A('      arrastrar o seleccionar aquí', id='aviso-carga-enunciado', style={'color': '#3FF43F', 'fontSize': 13})
            ], style={'fontSize': 14}),
            multiple=False
        ),
        html.A('', id='aviso-error-enunciado', style={'color': 'red', 'fontSize': 13}),
        html.Br(),
        html.Br(),
        dbc.Stack([
            dbc.Button('Analizar',
                       id='button-analizar',
                       color="success",
                       className="mb-3"
            ),#, style={'display': 'inline-block'}
            dbc.Button('Borrar datos cargados',
                       id='button-borrar',
                       color="secondary",
                       className="mb-3"
            ),#, style={'display': 'inline-block'}
        ],
        direction="horizontal",
        gap=4),
        html.A('', id='aviso-advertencia', style={'color': 'yellow', 'fontSize': 16}),
        html.A('', id='aviso-error-no-carga-informes', style={'color': 'red', 'fontSize': 16}),
        html.Hr(),
        ############ a partir de aca van a estar los dos tabs con su contenido
        dbc.Tabs(
            [
                dbc.Tab(tab1_content, label="Ranking", active_tab_style={"textTransform": "uppercase"}),
                dbc.Tab(tab2_content, label="Grupos", active_tab_style={"textTransform": "uppercase"})
            ],
            style={'borderColor': '#4682B4', 'borderWidth': '2px'}  # Ajusta 'tu_color_preferido' al color que desees
        ),
        # dcc.Store para almacenar las variables intermedias
        dcc.Store(id='output-json-informes'),
        dcc.Store(id='json-enunciado'),
        dcc.Store(id='output-json-informes-sin-enunciado')
    ]
)


######################## FUNCIONES GLOBALES

# Define regular expression pattern to extract student name and ID from folder names
folder_pattern = re.compile(r'(.+)_([0-9]+)_.*')

# preprocesamiento primario de los textos
def preprocesamiento_primario_textos(text):
    
    # \n es el fin de una linea e texto
    # \r\n es el inicio de una linea de texto
    # \x0c es un salto de página
    # \n̂ es un salto de línea y un posible cambio de indentación en el código fuente
    
    # Remuevo los caracteres especiales y reemplazo por espacios " "
    text = text.replace(r'\n{1,}|\r{1,}|\x0c{1,}|\n̂{1,}', ' ')
    
    # Uso unicode que normaliza el texto, por ejemplo remueve tildes
    #text = unidecode(text) # ESTO VA A SER MEJOR HACERLO APARTE A LO ULTIMO PARA QUE NO AFECTE LA LEMATIZACIÓN
    
    # Convierto todas las palabras a minúscula
    text = text.lower()
    
    # Eliminar espacios en blanco al principio y al final
    text = text.strip()
    
    # Remuevo signos de puntuación
    #text = re.sub(r'', ' ', text)
    
    #Remuevo cualquier caracter que no sea letra, numero o espacio en blanco
    #text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remuevo espacios en blanco de más entre palabras
    text = re.sub(r'\s+', ' ', text)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    return text

def preprocesamiento_enunciado(enunciado):
    
    # Los inicio de parrafos los cambio por parrafo_nuevo para despues tokenizar
    enunciado = enunciado.replace('\r\n', ' parrafo_nuevo ')
    
    # Remuevo los caracteres especiales y reemplazo por espacios en blanco
    enunciado = enunciado.replace('\x0c', ' ')
    enunciado = enunciado.replace('\n', ' ')
    enunciado = enunciado.replace('\n̂', ' ')
    
    # Convierto todas las palabras a minúscula
    enunciado = enunciado.lower()
    
    # Eliminar espacios en blanco al principio y al final
    enunciado = enunciado.strip()
    
    # Remuevo espacios en blanco de más entre palabras
    enunciado = re.sub(r'\s+', ' ', enunciado)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    enunciado = re.sub(r'\s+([.,;:])', r'\1', enunciado)
    
    # normalizo el texto
    #enunciado = unidecode(enunciado) # ESTO NO HARIA FALTA PARA EL ENUNCIADO PORQUE TODAVIA NO LO HAGO PARA LOS TEXTOS
    
    return enunciado

def eliminar_tokens_una_letra_numero_items(tokens):
    """
    Elimina los tokens que sean una única letra o número o nada
    """
    return [token for token in tokens if len(token) > 1 or not token.isalnum()]

def eliminar_tokens_una_palaba(tokens):
    """
    Elimina los tokens que sean una única palabra
    """
    tokens_filtrados = []
    
    for token in tokens:
        if len(token.split()) > 1:
            tokens_filtrados.append(token)
    
    return tokens_filtrados

def eliminar_tokens_espacios(tokens):
    """
    Elimina los espacios en blanco al inicio y fin
    """
    tokens_sin_espacios = []
    
    for token in tokens:
        token = token.strip()
        tokens_sin_espacios.append(token)
    
    return tokens_sin_espacios

def preprocesamiento_secundario_textos(text):

    # Elimino cualquier caracter que no sea letra, espacio o punto # esto eliminaria numeros y signos de operaciones matemáticas
    text = re.sub('[^A-Za-z\sñáéíóúüÁÉÍÓÚÜ\.]+', '', text)
    
    # Eliminar espacios en blanco al principio y al final
    text = text.strip()

    # Remuevo espacios en blanco de más entre palabras
    text = re.sub(r'\s+', ' ', text)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    # Reemplazo varios puntos consecutivos por uno solo (estaría considerando que al llegar a una fórmula, la oración termina)
    text = re.sub('\.+', '.', text)
    
    return text

# Definir una función para lematizar y remover stopwords
def lemmatizar_y_remover_stopwords(text): # ACA LUEGO DE ESTO PASO A UNIDECODE PORQUE YA ESTARIA PREPROCESADO
    
    # Procesa el texto con spaCy
    doc = nlp_spacy(text)
    
    # Remueve las stopwords y lematiza
    tokens_procesados = [token.lemma_ for token in doc if not token.is_stop]
    
    # vuelvo a unir tokens con un espacio
    clean_text = " ".join(tokens_procesados)
    
    # Remuevo espacioas en blanco antes de los signos de puntuación
    clean_text = re.sub(r'\s+([.,;:])', r'\1', clean_text)
    
    # normalizo el texto con unidecode
    clean_text = unidecode(clean_text)
    
    # elimino las palabras "version" y "estudiantil" porque es lo que figura en las capturas de pantalla de InfoStat
    clean_text = clean_text.replace("version", "")
    clean_text = clean_text.replace("estudiantil", "")
    
    # elimino palabras conformadas únicamente por una letra que no esté en el listado de palabras de una letra (esto es por si quedaron caracteres basura sueltos, puede haber por las formulas matematicas)
    palabras = clean_text.split()
    palabras_filtradas = [palabra for palabra in palabras if (len(palabra) > 1 or palabra.lower() in ['a', 'e', 'y', 'o', 'u'])]
    # uno las palabras filtradas de nuevo en un solo texto
    clean_text = ' '.join(palabras_filtradas)
    
    return clean_text

# Inicializar el tokenizador de oraciones
tokenizador_oraciones = re.compile(r'\.')
def tokenizador_textos_oraciones(text):
    
    # tokenizo en oraciones ante la presencia de un punto
    tokens_oraciones = tokenizador_oraciones.split(text)
    
    # elimino tokens que sean una letra o número
    tokens_oraciones = eliminar_tokens_una_letra_numero_items(tokens_oraciones)
    
    # elimino tokens que sean una sola palabra
    tokens_oraciones = eliminar_tokens_una_palaba(tokens_oraciones)
    
    # elimno tokens que sean espacios vacios
    tokens_oraciones = eliminar_tokens_espacios(tokens_oraciones)
    
    return tokens_oraciones

# ahora que ya realice la tokenización de oraciones, puedo sacarle los puntos al texto
def ultimo_preprocesamiento_elimino_puntos(text):
    
    # elimino los puntos
    clean_text = re.sub(r'\.', '', text)
    
    # DE VUELTA (PORQUE AL SACAR PUNTOS VUELVE A HABER) elimino palabras conformadas únicamente por una letra que no esté en el listado de palabras de una letra (esto es por si quedaron caracteres basura sueltos, puede haber por las formulas matematicas)
    palabras = clean_text.split()
    palabras_filtradas = [palabra for palabra in palabras if (len(palabra) > 1 or palabra.lower() in ['a', 'e', 'y', 'o', 'u'])]
    # uno las palabras filtradas de nuevo en un solo texto
    clean_text = ' '.join(palabras_filtradas)
    
    return clean_text

# incluyo tokenización de palabras
def tokenizador_palabras(text):
    # remuevo los puntos
    text = text.replace(".", "")
    # tokenizo
    tokens = word_tokenize(text.lower(), language='spanish')
    return tokens

# incluyo tokenizaciones de n gramas con n=5, n=7 y n=9
def tokenizador_ngramas(tokens_palabras, n):
    tokens_ngrams = list(ngrams(tokens_palabras, n))
    tokens_ngrams = [' '.join(list(tupla)) for tupla in tokens_ngrams]
    return tokens_ngrams

# defino funcion para generar vectores Doc2Vec
def vectorizar_texto_Doc2Vec(tokens_palabras):
    
    vector = modelo_doc2vec.infer_vector(tokens_palabras)
    
    vector = vector.reshape(1, -1) # porque despues calculo la similitud de coseno
    
    return vector

# Jaccard: cantidad de coincidencias (intersección) entre tokens/union total de tokens 
def jaccard_similarity(tokens1, tokens2):
    set1=set(tokens1)
    set2=set(tokens2)
    largo_interseccion = len(set1.intersection(set2))
    largo_union = len(set1.union(set2))
    return largo_interseccion / largo_union






######################## FUNCIONES CALLBACK (REACTIVIDAD)

# PROCESAMIENTO DE INFORMES, AVISO DE ZIP CARGADO
@app.callback(
    Output(component_id='output-json-informes', component_property='data'),
    Output(component_id='aviso-carga-zip', component_property='children'),
    Output(component_id='aviso-error', component_property='children'),
    Input(component_id='upload-informes', component_property='contents'),
    Input(component_id='button-borrar', component_property='n_clicks'),
    prevent_initial_call=True
)

def preprocesamiento_avisoCarga(contents, n_clicks_borrar):
    
    # CALLBACK LECTURA DE INFORMES
    if contents is None:
        return None, '      arrastrar o seleccionar aquí', ""
    
    else:
        # Decodifica el contenido del archivo ZIP
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:    
            zip_file = zipfile.ZipFile(io.BytesIO(decoded))
        except:
            json_vacio = json.dumps({})
            return json_vacio, '      arrastrar o seleccionar aquí', 'ERROR: ¡El archivo cargado no corresponde a un formato ZIP!'
        
        # Create an empty list to store the dictionaries
        results_list = []
        
        # Loop over all the files in the zip file
        for file_name in zip_file.namelist():
            
            # Check if the file is a PDF
            if file_name.endswith('.pdf'):
                
                # Extract the student name and ID from the folder name
                folder_name = os.path.dirname(file_name)
                match = folder_pattern.match(folder_name)
                if match:
                    name, id = match.groups()
                else:
                    json_vacio = json.dumps({})
                    return json_vacio, '      arrastrar o seleccionar aquí', 'ERROR: ¡El archivo cargado no posee la estructura de entrega de moodle correspondiente, revisar el formato de las carpetas del mismo!'
                
                # inicializo la lista de paginas del pdf
                paginas_pdf = []
                
                # Extract the text from the PDF
                with zip_file.open(file_name, mode = 'r') as pdf_file:
                    pdf = pdftotext.PDF(pdf_file)
                    for page in pdf:
                        paginas_pdf.append(page)

                # uno todas las páginas en un único string
                text = '\n'.join(paginas_pdf)
                    
                    
                    # otra forma con pypdf2
                    #pdf_data = pdf_file.read()
                    #pdf_buffer = BytesIO(pdf_data) #es un binary buffer para leer en modo binario porque son PDFs (sino debería pasar a txt)
                    #pdf_reader = PdfReader(pdf_file)
                    #owner = pdf_reader.metadata.author
                    #text = ''
                    #for page_num in range(len(pdf_reader.pages)):
                    #    page = pdf_reader.pages[page_num]
                    #    text += page.extract_text()
                
                # Extract the owner from the PDF metadata
                #with zip_file.open(file_name) as pdf_file:
                #    pdf_reader = PdfReader(pdf_file)
                #    owner = pdf_reader.metadata().author
                
                
                # Add a new dictionary to the list
                results_list.append({'Name': name,
                                     'ID': id,
                                     'Text': text})#,
                                     #'Owner': owner})
                
                # Add a new row to the results DataFrame
                #results_df = results_df.append({'Name': name,
                #                                'ID': id,
                #                                'Text': text,
                #                                'Owner': owner}, ignore_index=True)
                    
                    
        # concateno todos los diccionarios en un df
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in results_list], ignore_index=True)
    
        # Aplico la funciones de preprocesamiento a la columna "Texto" del df
        df['Text'] = df['Text'].apply(preprocesamiento_primario_textos)
        
        
        return df.to_json(), ' informes cargados', ""



# PROCESAMIENTO EN CASO DE QUE SE CARGUE EL ENUNCIADO, SE PROCESA EL MISMO Y SE QUITA DE LOS INFORMES
@app.callback(
    Output(component_id='output-json-informes-sin-enunciado', component_property='data'),
    Output(component_id='aviso-carga-enunciado', component_property='children'),
    Output(component_id='aviso-error-enunciado', component_property='children'),
    State(component_id='output-json-informes', component_property='data'),
    Input(component_id='upload-enunciado', component_property='contents'),
    Input(component_id='button-borrar', component_property='n_clicks'),
    prevent_initial_call=True
)

def preprocesamiento_eliminar_enunciado(json_data, enunciado_contents, n_clicks_borrar):
    
    if enunciado_contents is None:
        return None, '      arrastrar o seleccionar aquí', ""
        
    else:
        if json_data is None:
            raise PreventUpdate
        
        else:
            df = pd.read_json(json_data)
            
            # enunciado
            try:
                enunciado_data = base64.b64decode(enunciado_contents.split(",")[1])
                with io.BytesIO(enunciado_data) as pdf_buffer:
                    pdf = pdftotext.PDF(pdf_buffer)
                    paginas_enunciado = []
                    for page in pdf:
                        paginas_enunciado.append(page)
                # uno todas las páginas en un único string
                enunciado = '\n'.join(paginas_enunciado)
            
            except pdftotext.Error:
                #json_vacio = json.dumps({})
                json_vacio = None
                return json_vacio, '      arrastrar o seleccionar aquí', "ERROR: ¡El archivo cargado no corresponde a un formato PDF!"
            
            # preproceso el enunciado
            enunciado = preprocesamiento_enunciado(enunciado)
            
            # Tokenizo el enunciado en oraciones y parrafos
            tokenizador_oraciones_parrafos = re.compile(r'parrafo_nuevo|\n{1,}|[.:!?•]\s+|[a-zA-Z]\)')
            enunciado_oraciones_parrafos = tokenizador_oraciones_parrafos.split(enunciado)
            
            # elimno aquellos tokens que son items, es decir una letra o un número
            enunciado_oraciones_parrafos = eliminar_tokens_una_letra_numero_items(enunciado_oraciones_parrafos)
            
            # elimino aquellos tokens que son una única palabra porque podría generar eliminar palabras no debidas en el informe
            enunciado_oraciones_parrafos = eliminar_tokens_una_palaba(enunciado_oraciones_parrafos)

            # Eliminar espacios en blanco al principio y al final
            enunciado_oraciones_parrafos = eliminar_tokens_espacios(enunciado_oraciones_parrafos)
            
            # Ahora elimino estos tokens almacenados en enunciado_oraciones_parrafos de los textos de los informes
            for i, row in df.iterrows():
                for token in enunciado_oraciones_parrafos:
                    df.at[i, 'Text'] = df.at[i, 'Text'].replace(token, '')
            
            
            return df.to_json(), ' enunciado cargado y eliminado de cada informe', ""


# PROCESAMIENTOS SECUNDARIOS DE INFORMES, CALCULO DE METRICAS, CLASIFICADOR DE PARES Y AGRUPADOR
@app.callback(
    Output(component_id='output-tabla-ranking', component_property='children'),
    Output(component_id='output-tabla-grupos', component_property='children'),
    Output(component_id='aviso-advertencia', component_property='children'),
    Output(component_id='aviso-error-no-carga-informes', component_property='children'),
    Input(component_id='button-analizar', component_property='n_clicks'),
    State(component_id='output-json-informes-sin-enunciado', component_property='data'),
    State(component_id='output-json-informes', component_property='data'),
    Input(component_id='button-borrar', component_property='n_clicks'),
    prevent_initial_call=True
)

def preprocesamiento_secundario_scoring_clasificador_agrupador(n_clicks, json_data_SE, json_data_CE, n_clicks_borrar):
    
    if n_clicks is None:
        return None, None, "", ""
    
    else:
        if json_data_SE is None:
            if json_data_CE is None:
                return None, None, "", "ERROR: ¡El archivo de informes no fue cargado o posee un formato incorrecto!"
            else:
                df = pd.read_json(json_data_CE)
                df['Text'] = df['Text'].apply(preprocesamiento_secundario_textos)
                df['Text'] = df['Text'].apply(lemmatizar_y_remover_stopwords)
                #df['Tokens_oraciones'] = df['Text'].apply(tokenizador_textos_oraciones)
                df['Text'] = df['Text'].apply(ultimo_preprocesamiento_elimino_puntos)
                
                # armo un df único con cada tokenización
                df_tokens = pd.DataFrame({
                    'Nombre': df['Name'],
                    'ID': df['ID'],
                    'Texto': df['Text']#,
                    #'Tokens_oraciones': df['Tokens_oraciones']
                })
                
                # tokenizaciones
                df_tokens['Tokens_palabras'] = df_tokens['Texto'].apply(tokenizador_palabras)
                df_tokens['Tokens_3gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 3))
                #df_tokens['Tokens_7gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 7))
                
                # TFIDF vectorizer
                # paso los textos preprocesados a una lista
                lista_textos = list(df_tokens['Texto'])
    
                #### para palabras (1gramas), 3gramas, 7gramas # 3 y 7 gramas, segun el analisis de significancia solo usaremos el de 7 gramas
                for k in [7]:
                    # inicializo el vectorizador
                    tfidf_vectorizer = TfidfVectorizer(ngram_range=(k, k))
    
                    # ajuste de la vectorización en base a los textos
                    textos_tfidf_vectorizer_transformados = tfidf_vectorizer.fit_transform(lista_textos)
                    
                    # veo los nombreS de los tokens unicos identificados
                    tfidf_vectorizer.vocabulary_
                    
                    # agrego la columna con el texto vectorizado # LOS AGREGO COMO LISTAS PERO DESPUES PARA USARLOS HAY QUE PASARLOS DE VUELTA A ARRAY
                    if k==1:
                        df_tokens['vectorizado_TfidfVectorizer_palabras'] = list(textos_tfidf_vectorizer_transformados.toarray())
                    else:
                        df_tokens['vectorizado_TfidfVectorizer_'+str(k)+'gramas'] = list(textos_tfidf_vectorizer_transformados.toarray())
                
                # vectorizo Doc2Vec
                df_tokens['vectorizado_Doc2Vec'] = df_tokens['Tokens_palabras'].apply(vectorizar_texto_Doc2Vec)
                
                
                # calculo para todas las combinaciones Jaccard, Coseno TFIDF y Coseno Doc2Vec
                num_textos = len(df_tokens)
                similarity_list = []
                for i, j in combinations(range(num_textos), 2):
                    similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], 
                                            jaccard_similarity(set(df_tokens['Tokens_palabras'][i]), set(df_tokens['Tokens_palabras'][j])), 
                                            jaccard_similarity(set(df_tokens['Tokens_3gramas'][i]), set(df_tokens['Tokens_3gramas'][j])),
                                            #jaccard_similarity(set(df_tokens['Tokens_7gramas'][i]), set(df_tokens['Tokens_7gramas'][j])),
                                            #jaccard_similarity(set(df_tokens['Tokens_oraciones'][i]), set(df_tokens['Tokens_oraciones'][j])),
                                            cosine_similarity([df_tokens['vectorizado_TfidfVectorizer_7gramas'][i]], [df_tokens['vectorizado_TfidfVectorizer_7gramas'][j]])[0][0],
                                            cosine_similarity(df_tokens['vectorizado_Doc2Vec'][i], df_tokens['vectorizado_Doc2Vec'][j])[0][0]])
    
                df_final = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_palabras',
                                                                  'Similitud_Jaccard_3gramas',
                                                                  #'Similitud_Jaccard_7gramas',
                                                                  #'Similitud_Jaccard_oraciones',
                                                                  'Similitud_Coseno_TFIDFVec_7gramas',
                                                                  'Similitud_Coseno_Doc2Vec'])
                
                df_clustering = df_final # esto es para usarlo en el clustering
                
                ### CLASIFICADOR
                # Inicializar el escalador
                scaler = StandardScaler()

                # escalo las variables para predecir
                scores_final_scaled = scaler.fit_transform(df_final.drop(['ID1', 'ID2'], axis=1))            

                # predicciones
                gb_predictions = pd.DataFrame(gb_model.predict(scores_final_scaled), columns=['Predichos'])
                gb_predictions = gb_predictions.set_index(pd.Index(list(df_final.index)))
                #df_final_scaled = pd.DataFrame(df_final_scaled, columns=list(df_final.columns))
                df_final = pd.concat([df_final, 
                                      gb_predictions], 
                                     axis=1)

                # me quedo solo con los pares que son clasificados como alta similitud (=1)
                df_final = df_final[df_final['Predichos']==1]
                df_final = df_final.drop('Predichos', axis=1)

                # importancia de las variables
                ponderaciones_importancia_gini = gb_model.feature_importances_

                # Calcular el promedio ponderado y agregarlo como una nueva columna al DataFrame
                df_final['Score de Similitud'] = (df_final[['Similitud_Jaccard_palabras', 'Similitud_Jaccard_3gramas', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']] * ponderaciones_importancia_gini).sum(axis=1)        
                df_final['Score de Similitud'] = df_final['Score de Similitud'].astype(float)
                
                # Ordena el DataFrame por la columna 'Score de Similitud' de mayor a menor
                df_final_sorted = df_final.sort_values(by='Score de Similitud', ascending=False)
                df_final_sorted = df_final_sorted[['ID1', 'ID2', 'Score de Similitud']]
                
                # lista con los IDs correspondientes a alta similitud para despues filtrar la tabla de grupos
                lista_IDs_alta_similitud = list(set(list(df_final_sorted['ID1'])+list(df_final_sorted['ID2'])))
                
                # redondeo el score para el ranking
                df_final_sorted['Score de Similitud'] = df_final_sorted['Score de Similitud'].round(3)
                
                tabla_ranking = dbc.Table.from_dataframe(df_final_sorted, striped=True, bordered=True, hover=True)
                
                ### CLUSTERING JERARQUICO
                # convertir el df de pares con similitudes a una matriz de distacias
                df_clustering['Distancia'] = 1-(df_clustering[['Similitud_Jaccard_palabras', 'Similitud_Jaccard_3gramas', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']] * ponderaciones_importancia_gini).sum(axis=1)  
                df_clustering = df_clustering[['ID1', 'ID2', 'Distancia']]
                
                # Reordena los índices para asegurarte de que los mismos IDs estén en las mismas posiciones
                idx = sorted(set(df_clustering['ID1'].unique()) | set(df_clustering['ID2'].unique()))
                
                # Crea la tabla pivote asegurándote de que los índices coincidan
                pivot_table = df_clustering.pivot_table(index='ID1', columns='ID2', values='Distancia', fill_value=0)
                pivot_table = pivot_table.reindex(index=idx, columns=idx, fill_value=0)
                
                # Suma la tabla pivote con su traspuesta para hacerla simétrica
                dist_matrix = pivot_table.values + pivot_table.T.values
                
                # Obtén los nombres de las columnas para usarlos como nombres de las filas y columnas de la matriz
                nombres_columnas = pivot_table.columns
                
                # Crea un DataFrame con la matriz y los nombres de las columnas como índices
                df_matrix = pd.DataFrame(dist_matrix, index=nombres_columnas, columns=nombres_columnas)

                # IDs ordenados de la matriz
                IDs_ordenados_matrix = df_matrix.columns

                # matriz de distancias como array 2D
                array_matrix = df_matrix.values

                # clustering jerarquico con linkage WARD
                Z = hierarchy.linkage(array_matrix, method='ward')
                
                # Definir la distancia límite
                t = 1.3

                # Realizar el corte en la distancia límite
                clusters = hierarchy.fcluster(Z, t, criterion='distance')
                
                # armo un df con los IDs y los grupos
                informes_clusters = pd.concat([pd.DataFrame(list(IDs_ordenados_matrix), columns=['IDs']), pd.DataFrame(clusters, columns=['cluster'])], axis=1)
                            
                # Agrupar por 'cluster' y contar el número de observaciones en cada grupo para quedarme solo con grupos de 3 o más
                grupos_con_observaciones = informes_clusters.groupby('cluster').filter(lambda x: len(x) >= 3)
                
                # Obtener los grupos con 2 o más observaciones y los IDs correspondientes
                grupos_con_observaciones_ids = grupos_con_observaciones.groupby('cluster')['IDs'].apply(list).to_frame()
                
                # me quedo solo con los grupos que tienen algun ID con alta similitud
                # Función para verificar si algún elemento de la lista está presente en lista_sin_duplicados
                def contiene_elemento_en_lista(ids_lista):
                    return any(id_ in lista_IDs_alta_similitud for id_ in ids_lista)
                
                # Filtrar el DataFrame utilizando la función y crear un nuevo DataFrame con las filas filtradas
                grupos_con_observaciones_ids = grupos_con_observaciones_ids[grupos_con_observaciones_ids['IDs'].apply(lambda x: contiene_elemento_en_lista(x))]
                
                # Función para convertir una lista de IDs en un string separado por comas
                def ids_a_string(ids):
                    return ', '.join(map(str, ids))

                # Aplicar la función a la columna 'IDs' para convertir listas en strings
                grupos_con_observaciones_ids['IDs_string'] = grupos_con_observaciones_ids['IDs'].apply(ids_a_string)

                # Crear un nuevo DataFrame con las columnas deseadas
                df_final_grupos = pd.DataFrame({
                    'Grupos de informes a revisar': [f'Grupo {chr(65 + i)}' for i in range(len(grupos_con_observaciones_ids))],
                    'IDs incluidos en este grupo': grupos_con_observaciones_ids['IDs_string']
                })
                
                ########################### ACA FALTA HACER ALGUN FILTRO DE ESTA TABLA PARA QUE SOLO MUESTRE LOS GRUPOS CON ALGUNO DE ALTA SIMILITUD
                
                # tabla de grupos
                tabla_grupos = dbc.Table.from_dataframe(df_final_grupos, striped=True, bordered=True, hover=True)
                
                return tabla_ranking, tabla_grupos, "¡Advertencia! No se cargó un enunciado en formato válido para ser eliminado de los informes. Si es intencional desestime la advertencia.", ""
        
        else:
            df = pd.read_json(json_data_SE)
            df['Text'] = df['Text'].apply(preprocesamiento_secundario_textos)
            df['Text'] = df['Text'].apply(lemmatizar_y_remover_stopwords)
            #df['Tokens_oraciones'] = df['Text'].apply(tokenizador_textos_oraciones)
            df['Text'] = df['Text'].apply(ultimo_preprocesamiento_elimino_puntos)
            
            # armo un df único con cada tokenización
            df_tokens = pd.DataFrame({
                'Nombre': df['Name'],
                'ID': df['ID'],
                'Texto': df['Text']#,
                #'Tokens_oraciones': df['Tokens_oraciones']
            })
            
            # tokenizaciones
            df_tokens['Tokens_palabras'] = df_tokens['Texto'].apply(tokenizador_palabras)
            df_tokens['Tokens_3gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 3))
            #df_tokens['Tokens_7gramas'] = df_tokens['Tokens_palabras'].apply(lambda x: tokenizador_ngramas(x, 7))
            
            # TFIDF vectorizer
            # paso los textos preprocesados a una lista
            lista_textos = list(df_tokens['Texto'])

            #### para palabras (1gramas), 3gramas, 7gramas # 3 y 7 gramas, segun el analisis de significancia solo usaremos el de 7 gramas
            for k in [7]:
                # inicializo el vectorizador
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(k, k))

                # ajuste de la vectorización en base a los textos
                textos_tfidf_vectorizer_transformados = tfidf_vectorizer.fit_transform(lista_textos)
                
                # veo los nombreS de los tokens unicos identificados
                tfidf_vectorizer.vocabulary_
                
                # agrego la columna con el texto vectorizado # LOS AGREGO COMO LISTAS PERO DESPUES PARA USARLOS HAY QUE PASARLOS DE VUELTA A ARRAY
                if k==1:
                    df_tokens['vectorizado_TfidfVectorizer_palabras'] = list(textos_tfidf_vectorizer_transformados.toarray())
                else:
                    df_tokens['vectorizado_TfidfVectorizer_'+str(k)+'gramas'] = list(textos_tfidf_vectorizer_transformados.toarray())
            
            # vectorizo Doc2Vec
            df_tokens['vectorizado_Doc2Vec'] = df_tokens['Tokens_palabras'].apply(vectorizar_texto_Doc2Vec)
            
            
            # calculo para todas las combinaciones Jaccard, Coseno TFIDF y Coseno Doc2Vec
            num_textos = len(df_tokens)
            similarity_list = []
            for i, j in combinations(range(num_textos), 2):
                similarity_list.append([df_tokens['ID'][i], df_tokens['ID'][j], 
                                        jaccard_similarity(set(df_tokens['Tokens_palabras'][i]), set(df_tokens['Tokens_palabras'][j])), 
                                        jaccard_similarity(set(df_tokens['Tokens_3gramas'][i]), set(df_tokens['Tokens_3gramas'][j])),
                                        #jaccard_similarity(set(df_tokens['Tokens_7gramas'][i]), set(df_tokens['Tokens_7gramas'][j])),
                                        #jaccard_similarity(set(df_tokens['Tokens_oraciones'][i]), set(df_tokens['Tokens_oraciones'][j])),
                                        cosine_similarity([df_tokens['vectorizado_TfidfVectorizer_7gramas'][i]], [df_tokens['vectorizado_TfidfVectorizer_7gramas'][j]])[0][0],
                                        cosine_similarity(df_tokens['vectorizado_Doc2Vec'][i], df_tokens['vectorizado_Doc2Vec'][j])[0][0]])

            df_final = pd.DataFrame(similarity_list, columns=['ID1', 'ID2', 'Similitud_Jaccard_palabras',
                                                              'Similitud_Jaccard_3gramas',
                                                              #'Similitud_Jaccard_7gramas',
                                                              #'Similitud_Jaccard_oraciones',
                                                              'Similitud_Coseno_TFIDFVec_7gramas',
                                                              'Similitud_Coseno_Doc2Vec'])

            df_clustering = df_final # esto es para usarlo en el clustering
            
            ### CLASIFICADOR
            # Inicializar el escalador
            scaler = StandardScaler()

            # escalo las variables para predecir
            scores_final_scaled = scaler.fit_transform(df_final.drop(['ID1', 'ID2'], axis=1))            

            # predicciones
            gb_predictions = pd.DataFrame(gb_model.predict(scores_final_scaled), columns=['Predichos'])
            gb_predictions = gb_predictions.set_index(pd.Index(list(df_final.index)))
            #df_final_scaled = pd.DataFrame(df_final_scaled, columns=list(df_final.columns))
            df_final = pd.concat([df_final, 
                                  gb_predictions], 
                                 axis=1)

            # me quedo solo con los pares que son clasificados como alta similitud (=1)
            df_final = df_final[df_final['Predichos']==1]
            df_final = df_final.drop('Predichos', axis=1)

            # importancia de las variables
            ponderaciones_importancia_gini = gb_model.feature_importances_

            # Calcular el promedio ponderado y agregarlo como una nueva columna al DataFrame
            df_final['Score de Similitud'] = (df_final[['Similitud_Jaccard_palabras', 'Similitud_Jaccard_3gramas', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']] * ponderaciones_importancia_gini).sum(axis=1)        
            df_final['Score de Similitud'] = df_final['Score de Similitud'].astype(float)
            
            # Ordena el DataFrame por la columna 'Score de Similitud' de mayor a menor
            df_final_sorted = df_final.sort_values(by='Score de Similitud', ascending=False)
            df_final_sorted = df_final_sorted[['ID1', 'ID2', 'Score de Similitud']]
            
            # lista con los IDs correspondientes a alta similitud para despues filtrar la tabla de grupos
            lista_IDs_alta_similitud = list(set(list(df_final_sorted['ID1'])+list(df_final_sorted['ID2'])))
            
            # redondeo el score para el ranking
            df_final_sorted['Score de Similitud'] = df_final_sorted['Score de Similitud'].round(3)
            
            tabla_ranking = dbc.Table.from_dataframe(df_final_sorted, striped=True, bordered=True, hover=True)
            
            ### CLUSTERING JERARQUICO
            # convertir el df de pares con similitudes a una matriz de distacias
            df_clustering['Distancia'] = 1-(df_clustering[['Similitud_Jaccard_palabras', 'Similitud_Jaccard_3gramas', 'Similitud_Coseno_TFIDFVec_7gramas', 'Similitud_Coseno_Doc2Vec']] * ponderaciones_importancia_gini).sum(axis=1)
            df_clustering = df_clustering[['ID1', 'ID2', 'Distancia']]
            
            # Reordena los índices para asegurarte de que los mismos IDs estén en las mismas posiciones
            idx = sorted(set(df_clustering['ID1'].unique()) | set(df_clustering['ID2'].unique()))
            
            # Crea la tabla pivote asegurándote de que los índices coincidan
            pivot_table = df_clustering.pivot_table(index='ID1', columns='ID2', values='Distancia', fill_value=0)
            pivot_table = pivot_table.reindex(index=idx, columns=idx, fill_value=0)
            
            # Suma la tabla pivote con su traspuesta para hacerla simétrica
            dist_matrix = pivot_table.values + pivot_table.T.values
            
            # Obtén los nombres de las columnas para usarlos como nombres de las filas y columnas de la matriz
            nombres_columnas = pivot_table.columns
            
            # Crea un DataFrame con la matriz y los nombres de las columnas como índices
            df_matrix = pd.DataFrame(dist_matrix, index=nombres_columnas, columns=nombres_columnas)

            # IDs ordenados de la matriz
            IDs_ordenados_matrix = df_matrix.columns

            # matriz de distancias como array 2D
            array_matrix = df_matrix.values

            # clustering jerarquico con linkage WARD
            Z = hierarchy.linkage(array_matrix, method='ward')
            
            # Definir la distancia límite
            t = 1.3

            # Realizar el corte en la distancia límite
            clusters = hierarchy.fcluster(Z, t, criterion='distance')
            
            # armo un df con los IDs y los grupos
            informes_clusters = pd.concat([pd.DataFrame(list(IDs_ordenados_matrix), columns=['IDs']), pd.DataFrame(clusters, columns=['cluster'])], axis=1)
                        
            # Agrupar por 'cluster' y contar el número de observaciones en cada grupo para quedarme solo con grupos de 3 o más
            grupos_con_observaciones = informes_clusters.groupby('cluster').filter(lambda x: len(x) >= 3)
            
            # Obtener los grupos con 2 o más observaciones y los IDs correspondientes
            grupos_con_observaciones_ids = grupos_con_observaciones.groupby('cluster')['IDs'].apply(list).to_frame()
            
            # me quedo solo con los grupos que tienen algun ID con alta similitud
            # Función para verificar si algún elemento de la lista está presente en lista_sin_duplicados
            def contiene_elemento_en_lista(ids_lista):
                return any(id_ in lista_IDs_alta_similitud for id_ in ids_lista)
            
            # Filtrar el DataFrame utilizando la función y crear un nuevo DataFrame con las filas filtradas
            grupos_con_observaciones_ids = grupos_con_observaciones_ids[grupos_con_observaciones_ids['IDs'].apply(lambda x: contiene_elemento_en_lista(x))]
            
            # Función para convertir una lista de IDs en un string separado por comas
            def ids_a_string(ids):
                return ', '.join(map(str, ids))

            # Aplicar la función a la columna 'IDs' para convertir listas en strings
            grupos_con_observaciones_ids['IDs_string'] = grupos_con_observaciones_ids['IDs'].apply(ids_a_string)

            # Crear un nuevo DataFrame con las columnas deseadas
            df_final_grupos = pd.DataFrame({
                'Grupos de informes a revisar': [f'Grupo {chr(65 + i)}' for i in range(len(grupos_con_observaciones_ids))],
                'IDs incluidos en este grupo': grupos_con_observaciones_ids['IDs_string']
            })
            
            # tabla de grupos
            tabla_grupos = dbc.Table.from_dataframe(df_final_grupos, striped=True, bordered=True, hover=True)
            
            return tabla_ranking, tabla_grupos, "", ""


@app.callback(
    Output(component_id='upload-informes', component_property='contents'),
    Output(component_id='upload-enunciado', component_property='contents'),
    Output(component_id='button-analizar', component_property='n_clicks'),
    Input(component_id='button-borrar', component_property='n_clicks'),
    prevent_initial_call=True
)
def reset(n_clicks_borrar):
    return None, None, None

######################## URL

if __name__ == '__main__':
    app.run_server(debug=True, port = 8080) #http://127.0.0.1:8080/
