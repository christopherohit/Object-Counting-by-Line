import argparse
import os
import platform
from pathlib import Path

import torch
import sys
sys.path.append('yolov9/')
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from yolov9.utils.general import (LOGGER, check_img_size, increment_path, non_max_suppression, scale_boxes, xyxy2xywh, check_file,
                                  check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args,
                                  strip_optimizer, xyxy2xywh)

from yolov9.utils.plots import Annotator, colors, save_one_box
from yolov9.utils.torch_utils import select_device, smart_inference_mode
# from ultralytics.solutions import object_counter
import cv2
from ultralytics import YOLO
from src import solutions

# Load the pre-trained YOLOv8 model
model = YOLO("model/yolov8n.pt")

path = input("Path of image: ")
name_file = path.split('/')[-1].split('.')[0]
# Open the video file
cap = cv2.VideoCapture(filename=path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
# line_points = [(280, 200), (700, 180)]
line_points = [(480, 200), (900, 180)]
classes_to_count = [0]

classes_names = {0: "person"}  # example class names

# Init Object Counter
counter_remake = solutions.ObjectCounter(view_img=True,
                 reg_pts=line_points,
                 classes_names=classes_names,
                 draw_tracks=True,
                 line_thickness=2)

# Open log file for writing
log_file = open(f"logs/{name_file}.txt", "w")
video_writer = cv2.VideoWriter(f"result/result_{name_file}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
count = 0

# counter = solutions.ObjectCounter(
#     view_img=True,  # Display the image during processing
#     reg_pts=line_points,  # Region of interest points
#     classes_names=model.names,  # Class names from the YOLO model
#     draw_tracks=True,  # Draw tracking lines for objects
#     line_thickness=2,  # Thickness of the lines drawn
# )

# Process video frames in a loop
while cap.isOpened():
    count = count + 1
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking on the current frame, filtering by specified classes
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    # Use the Object Counter to count objects in the frame and get the annotated image
    im0 = counter_remake.start_counting(im0, tracks)

    for track in tracks:
        
        result = track.boxes.cpu().numpy()
        track_id = result.id
        bbox = result.xyxy
        class_id = result.cls
        confidence = result.conf

        ####################################
        # frame_in = result.frame_in #(int)
        # frame_out = result.frame_out #(int)
        # total_in = result.total_in #(int)
        # total_out = result.total_out #(int)
        #########################################
        log_file.write(f"{count}|{track_id}|{class_id}|{bbox}|{confidence}")
        log_file.write("\n")

    # Write the annotated frame to the output video
    video_writer.write(im0)

# Release the video capture and writer objects
cap.release()
video_writer.release()
