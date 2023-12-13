from queue import Queue
from threading import Thread
import os
import cv2
import json
from shapely import Point
from shapely.geometry import Polygon
from ultralytics import YOLO
import numpy as np


def create_polygon_mask(frame, polygons, color):
    frame_copy = frame.copy()
    cv2.polylines(frame_copy, [np.array(polygons)], isClosed=True, color=color, thickness=2)
    return frame_copy


def process_video(video_path, polygon_path, yolo_model_path, results, visualize):
    cap = cv2.VideoCapture(video_path)
    # Load stand boundaries from JSON
    with open(polygon_path, 'r') as f:
        stand_boundaries = json.load(f)

    model = YOLO(yolo_model_path) if yolo_model_path else None
    frame_count = 0
    intersection_frames = 0
    start_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Extract video name from the full path
        video_name = os.path.basename(video_path)

        # Check if stand boundaries are available for the current video
        if video_name in stand_boundaries:
            color = (255, 0, 0)
            polygons = stand_boundaries[video_name]
            roi_polygon = Polygon(polygons)

            if model:
                detections = model(frame, stream=True, conf=0.35)

                intersection_detected = False  # Flag to track if any intersection is detected in this frame

                for detection in detections:
                    boxes = detection.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        mid_point = Point(mid_x, mid_y)

                        if roi_polygon.intersects(mid_point):
                            print(f"Intersection with ROI on frame {frame_count}")
                            color = (0, 255, 0)
                            intersection_detected = True
                            intersection_frames += 1

                            if start_frame is None:
                                start_frame = frame_count

                if not intersection_detected and start_frame is not None:
                    if intersection_frames > 2:
                        results[video_name].append([start_frame, frame_count - 1])
                    start_frame = None
                    intersection_frames = 0

            frame_with_polygons = create_polygon_mask(frame.copy(), polygons, color=color)
            if visualize:
                cv2.imshow('Video', frame_with_polygons)
                cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()


def worker(video_queue, results, visualize):
    while True:
        video_path, polygon_path,  yolo_model_path = video_queue.get()
        if video_path is None:
            break
        process_video(video_path, polygon_path, yolo_model_path, results, visualize)
        video_queue.task_done()


def process_video_queue(video_folder, polygon_path, yolo_model_path, visualize):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(video_files)

    # Create a dictionary to store results
    results = {video_name: [] for video_name in video_files}

    # Create a queue for processing
    video_queue = Queue()

    # Fill the queue with video files
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_queue.put((video_path, polygon_path, yolo_model_path))

    # Define the worker function for processing the queue
    num_threads = min(len(video_files), 1)  # Specify the desired number of threads
    threads = []

    for _ in range(num_threads):
        thread = Thread(target=worker, args=(video_queue, results, visualize))
        thread.start()
        threads.append(thread)

    # Wait for the queue to be processed
    video_queue.join()

    # Stop the threads
    for _ in range(num_threads):
        video_queue.put((None, None, None, None))  # Add signaling tasks to terminate threads

    for thread in threads:
        thread.join()

    # Save the results as a JSON file
    with open('output_results.json', 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video folder and detect vehicles.")
    parser.add_argument("-video_folder", type=str, help="Path to the folder containing video files")
    parser.add_argument("-polygon_path", default='bin/polygons/polygons.json', type=str,
                        help="Path to JSON with stand boundaries")

    parser.add_argument("-yolo_model_path", default='cars.pt', type=str,
                        help="Path to YOLO model weights and configuration files")
    parser.add_argument("-visualize", action="store_true", help="Enable visualization")

    args = parser.parse_args()

    process_video_queue(args.video_folder, args.polygon_path, args.yolo_model_path, args.visualize)
