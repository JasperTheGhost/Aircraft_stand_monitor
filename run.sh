VIDEO_FOLDER="bin/videos"
POLYGON_PATH="bin/polygons/polygons.json"
OUTPUT_FOLDER="./"
YOLO_MODEL_PATH="./cars.pt"

# Run the Python script
python main.py \
    -video_folder "$VIDEO_FOLDER" \
    -polygon_path "$POLYGON_PATH" \
    -output_folder "$OUTPUT_FOLDER" \
    -yolo_model_path "$YOLO_MODEL_PATH"