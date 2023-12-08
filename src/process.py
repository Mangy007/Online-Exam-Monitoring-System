import cv2
from PIL import Image
import tensorflow as tf
import numpy as np

from features import CLIPFeatureExtracter

import os

class Processor():
    
    def __init__(self):
        self.clip_feature_extractor = CLIPFeatureExtracter()

    def process_video(self, videos_directory_path: str, frames_interval: int, output_directory_path: str):
        video_files = os.listdir(videos_directory_path)
        video_files.sort()

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        for video_file in video_files:
            video_path = os.path.join(videos_directory_path, video_file)
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            fps = int(fps*frames_interval)
            # Initialize an empty list to store CLIP vision vectors
            clip_vision_vectors = []
            count = 0
            # Read frames from the video and process them
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                count +=1
                if(count % fps != 0): continue
                # Convert the frame to the required format (resize, normalize, etc.)
                frame = cv2.resize(frame, (224, 224))  # Resize to CLIP model input size
                frame = frame / 255.0  # Normalize pixel values to [0, 1]
                # Ensure the data type is uint8
                frame = (frame * 255).astype(np.uint8)
                # Convert the frame to a PIL Image
                frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Extract features using the feature extractor
                pooled_output = self.clip_feature_extractor.extract_features(frame_image)
                # Append the vision features to the list
                clip_vision_vectors.append(pooled_output)
            # Release the video capture object
            video.release()

            # Concatenate the vision vectors to a single tensor
            clip_vision_vectors = tf.stack(clip_vision_vectors, axis=0)

            # Save CLIP vision vectors into numpy array
            file_path = output_directory_path+video_file.split('.')[0]+'.npy'
            np.save(file_path, clip_vision_vectors)