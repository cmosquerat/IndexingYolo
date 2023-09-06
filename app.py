# Importamos las bibliotecas necesarias
import streamlit as st
from indexing_utp import *
import tempfile
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="best.pt",
    confidence_threshold=0.7,
    device="cpu", # or 'cuda:0'
)

# Establecemos el título de la aplicación Streamlit
st.title("Prueba Indexación + Detección UTP")

# Permitimos al usuario subir un archivo de video al servidor
uploaded_file = st.file_uploader("Sube un video", type=['mp4'])

# Creamos una lista de opciones de videos para seleccionar
video_options = ["Peleas.mp4","Armas.mp4","Disparos.mp4"]
# Permitimos al usuario seleccionar un video de prueba desde las opciones
selected_video = st.selectbox("O selecciona un video de prueba", video_options)

# Campo de entrada para que el usuario introduzca texto para buscar en el video
user_input = st.text_input("Introduce algún texto para buscar aquí:")

# Creamos un botón "Ejecutar" para iniciar el proceso de búsqueda
if st.button("Ejecutar"):
   # Verificamos que el campo de texto no esté vacío
   if len(user_input) != 0:
       # Si el usuario ha subido un archivo de video, lo usamos
       if uploaded_file is not None: 
           video = uploaded_file
           # Creamos un archivo temporal para guardar el video
           tfile = tempfile.NamedTemporaryFile(delete=False)
           tfile.write(video.read())
           location = tfile.name
       else:
           # Si no, usamos el video seleccionado de la lista de opciones
           video = selected_video
           location = video
       # Mostramos el video en la aplicación
       st.video(video)
       # Extraemos los fotogramas del video y obtenemos su tasa de fotogramas por segundo
       video_frames, fps = extract_frames(location, 15)
       # Generamos las características del video para la búsqueda
       video_features, model, device = generate_video_features(video_frames)
       # Buscamos en el video basándonos en el texto introducido por el usuario
       frames, seconds, fig = search_video(user_input, video_frames, video_features, model, device, 15, fps)
       # Mostramos un gráfico relacionado con los resultados de la búsqueda
       st.plotly_chart(fig)
       # Mostramos los fotogramas resultantes de la búsqueda
       for frame, seconds in zip(frames, seconds):
           result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height = 256,
                slice_width = 256,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2
            )
           result.export_visuals("Resultados/")
           
           st.image("Resultados/prediction_visual.png")
