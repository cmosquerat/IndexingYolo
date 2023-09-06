import cv2
from PIL import Image
import math
import numpy as np
import clip
import torch
import plotly.express as px
import datetime
from IPython.core.display import HTML

def extract_frames(video,N):
    # The frame images will be stored in video_frames
    video_frames = []

    # Open the video file
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)

    current_frame = 0
    while capture.isOpened():
        # Read the current frame
        ret, frame = capture.read()

        # Convert it to a PIL image (required for CLIP) and store it
        if ret == True:
            video_frames.append(Image.fromarray(frame[:, :, ::-1]))
        else:
            break

    # Skip N frames
    current_frame += N
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Print some statistics
    print(f"Frames extracted: {len(video_frames)}")
    return video_frames,fps

def generate_video_features(video_frames,batch_size=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # You can try tuning the batch size for very large videos, but it should usually be OK
    batch_size = 256
    batches = math.ceil(len(video_frames) / batch_size)

    # The encoded features will bs stored in video_features
    video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

    # Process each batch
    for i in range(batches):
        print(f"Processing batch {i+1}/{batches}")

        # Get the relevant frames
        batch_frames = video_frames[i*batch_size : (i+1)*batch_size]

        # Preprocess the images for the batch
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)

        # Encode with CLIP and normalize
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

        # Append the batch to the list containing all features
        video_features = torch.cat((video_features, batch_features))

    # Print some stats
    print(f"Features: {video_features.shape}")
    return video_features,model,device

def search_video(search_query, video_frames,video_features,model,device,N,fps, display_heatmap=True, display_results_count=3):


  # Encode and normalize the search query using CLIP
  with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

  # Compute the similarity between the search query and each frame using the Cosine similarity
  similarities = (100.0 * video_features @ text_features.T)
  values, best_photo_idx = similarities.topk(display_results_count, dim=0)

  # Display the heatmap
  
  fig = px.imshow(similarities.T.cpu().numpy(), height=50, aspect='auto', color_continuous_scale='viridis')
  fig.update_layout(coloraxis_showscale=False)
  fig.update_xaxes(showticklabels=False)
  fig.update_yaxes(showticklabels=False)
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
  

  frames = []
  seconds = []

  # Display the top 3 frames
  for frame_id in best_photo_idx:
    frames.append(video_frames[frame_id])

    # Find the timestamp in the video and display it
    seconds.append(round(frame_id.cpu().numpy()[0] * N / fps))
  
  return frames,seconds,fig
    
  