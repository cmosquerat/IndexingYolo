import streamlit as st
import cv2
import torch
import clip
from PIL import Image
import time
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from email.mime.image import MIMEImage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
import base64
import os
import pickle
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import json

# File to store the credentials
TOKEN_PICKLE_FILE = 'token.pickle'
CREDENTIALS_JSON_FILE = 'credentials.json'
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def load_credentials():
    if os.path.exists(TOKEN_PICKLE_FILE):
        with open(TOKEN_PICKLE_FILE, 'rb') as token:
            return pickle.load(token)
    return None

def save_credentials(creds):
    with open(TOKEN_PICKLE_FILE, 'wb') as token:
        pickle.dump(creds, token)

def get_gmail_service():
    creds = load_credentials()
    print(f"La credencial es válida? {creds.valid}")
    print(f"La credencial está expirada? {creds.expired}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_JSON_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        save_credentials(creds)
    return build('gmail', 'v1', credentials=creds)

# Function to load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

# Load configuration
config = load_config('config.json')

# Carga el modelo CLIP y la función de preprocesamiento
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def send_email(recipient_email, subject, body, image=None):
    service = get_gmail_service()

    message = MIMEMultipart()
    message['to'] = recipient_email
    message['subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    if image is not None:
        img = MIMEImage(image)
        message.attach(img)

    encoded_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    try:
        sent_message = service.users().messages().send(userId="me", body=encoded_message).execute()
        print(F'Sent message to {recipient_email} Message Id: {sent_message["id"]}')
    except HTTPError as error:
        print(F'An error occurred: {error}')

def get_classification(frame):
    image = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    texto = clip.tokenize(["violence", "pedestrian"]).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, texto)
        probabilidades = logits_per_image.softmax(dim=-1).cpu().numpy()
    return probabilidades[0][0], probabilidades[0][1]

st.title("Clasificación de Transmisión UTP")

nombre_camara = st.text_input("Nombre de la Cámara:", "")
ubicacion = st.text_input("Ubicación (Lat, Lon):", "")
correo = st.text_input("Correo Electrónico:", "")
url_rtsp = st.text_input("Introduce URL RTSP/RTMP:", "")

# Extraer latitud y longitud
lat, lon = [float(x) for x in ubicacion.split(",")] if ubicacion else (0, 0)

# Crear un DataFrame para almacenar los puntajes de violencia y las marcas de tiempo
df = pd.DataFrame(columns=["Marca de Tiempo", "Probabilidad de Violencia"])

# Configura los espacios reservados antes del bucle
col1, col2 = st.columns(2)

with col1:
    espacio_col1 = st.empty()
    espacio_col3 = st.empty()

with col2:
    espacio_col2 = st.empty()

espacio_grafico = st.empty()

# Initialize the map with a point
map_data = [{'lat': lat, 'lon': lon, 'text': nombre_camara, 'marker': {'color': 'green', 'size': 15}}]

map_fig = go.Figure(go.Scattermapbox(
    lat=[d['lat'] for d in map_data],
    lon=[d['lon'] for d in map_data],
    mode='markers+text',  # Add both markers and text
    marker=dict(
        size=[d['marker']['size'] for d in map_data],
        color=[d['marker']['color'] for d in map_data]
    ),
    text=[d['text'] for d in map_data],  # Set text for each point
    textposition="top center"  # Position the text above the marker
))

# Set up the layout to use OpenStreetMap
map_fig.update_layout(
    mapbox=dict(
        accesstoken=None,  # No access token required for OpenStreetMap
        center=dict(lat=lat, lon=lon),
        zoom=15,  # Adjust zoom level as needed
        style='open-street-map'
    ),
    margin={"r":0,"t":0,"l":0,"b":0}  # Optional: Adjust margins
)


map_placeholder = st.empty()
map_placeholder.plotly_chart(map_fig, use_container_width=True)

if len(url_rtsp) > 0:
    cap = cv2.VideoCapture(url_rtsp)
    cuenta_frames = 0
    frame_window = []
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Error al leer el frame de la transmisión RTSP.")
            break

        cuenta_frames += 1
        if cuenta_frames % 10 == 0:
            prob_violencia, prob_no_violencia = get_classification(frame)

            frame_window.append((prob_violencia, frame))
            if len(frame_window) > 20:
                frame_window.pop(0)

            for prob, frm in frame_window:
                if prob > 0.8:
                    # Convert frame to JPEG for email attachment
                    ret, buffer = cv2.imencode('.jpg', frm)
                    image_bytes = buffer.tobytes()
                    if correo:  # Assuming 'correo' is the email input from the user
                        send_email(
                            recipient_email=correo,
                            subject="¡Alerta! Violencia Detectada",
                            body=f"Alta probabilidad de violencia detectada en la cámara con nombre {nombre_camara} y ubicada en {lat}, {lon}. Por favor, revise la imagen adjunta ",
                            image=image_bytes
                        )
                    else:
                        st.warning("No recipient email provided.")

                    # Clear the frame window to avoid repeated emails
                    frame_window.clear()
                    break


            espacio_col1.write(f"Probabilidad de Violencia: {prob_violencia:.2f}")
            espacio_col3.write(f"Probabilidad Sin Violencia: {prob_no_violencia:.2f}")
            espacio_col2.image(frame, channels="BGR", use_column_width=True)

            # Añade la marca de tiempo y la probabilidad de violencia al DataFrame
            marca_tiempo = time.strftime("%Y-%m-%d %H:%M:%S")
            df = pd.concat([df, pd.DataFrame({"Marca de Tiempo": [marca_tiempo], "Probabilidad de Violencia": [prob_violencia]})], ignore_index=True)

            # Actualiza el gráfico dinámico de tiempo
            fig = px.line(df, x="Marca de Tiempo", y="Probabilidad de Violencia", title="Probabilidad de Violencia a lo Largo del Tiempo")
            espacio_grafico.plotly_chart(fig, use_container_width=True, height=200)

            # Update the map color based on violence probability
            new_color = f'rgb({255 * prob_violencia}, {255 * (1 - prob_violencia)}, 0)'

            map_data[0]['marker']['color'] = new_color
            map_fig.data[0].marker.color = [d['marker']['color'] for d in map_data]
            map_placeholder.plotly_chart(map_fig, use_container_width=True)

            time.sleep(0.1)

    cap.release()
