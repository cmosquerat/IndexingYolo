import streamlit as st
import cv2
import torch
import clip
from PIL import Image
import time
import pandas as pd
import plotly.express as px

# Carga el modelo CLIP y la función de preprocesamiento
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_classification(frame):
    image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    texto = clip.tokenize(["violence", "pedestrian"]).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, texto)
        probabilidades = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probabilidades[0][0], probabilidades[0][1]

st.title("Clasificación de Transmisión UTP")

url_rtsp = st.text_input("Introduce URL RTSP/RTMP:", "")

# Crea un DataFrame para almacenar los puntajes de violencia y las marcas de tiempo
df = pd.DataFrame(columns=["Marca de Tiempo", "Probabilidad de Violencia"])

# Configura los espacios reservados antes del bucle
col1, col2 = st.columns(2)

with col1:
    espacio_col1 = st.empty()
    espacio_col3 = st.empty()

with col2:
    espacio_col2 = st.empty()


espacio_grafico = st.empty()

if url_rtsp:
    cap = cv2.VideoCapture(url_rtsp)
    cuenta_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Error al leer el frame de la transmisión RTSP.")
            break

        cuenta_frames += 1
        if cuenta_frames % 10 == 0:
            prob_violencia, prob_no_violencia = get_classification(frame)

            espacio_col1.write(f"Probabilidad de Violencia: {prob_violencia:.2f}")

            # Añade la marca de tiempo y la probabilidad de violencia al DataFrame
            marca_tiempo = time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.concat([df, pd.DataFrame({"Marca de Tiempo": [marca_tiempo], "Probabilidad de Violencia": [prob_violencia]})], ignore_index=True)

            # Actualiza el gráfico dinámico de tiempo
            fig = px.line(df, x="Marca de Tiempo", y="Probabilidad de Violencia", title="Probabilidad de Violencia a lo Largo del Tiempo")
            espacio_grafico.plotly_chart(fig, use_container_width=True, height=200)

            espacio_col2.image(frame, channels="BGR", use_column_width=True, width=300)
            espacio_col3.write(f"Probabilidad Sin Violencia: {prob_no_violencia:.2f}")

            # Duerme por un breve momento para permitir que Streamlit actualice la UI
            time.sleep(0.1)

    cap.release()