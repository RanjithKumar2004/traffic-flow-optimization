from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained YOLO model
model = YOLO(r'/home/ranjith/Education/fyp-final/backend/models/best.pt')

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Vehicle area mapping
vehicle_area = {
    'car': 50,
    'truck': 100,
    'bus': 80,
    'auto': 30,
    'two-wheeler': 20
}

def calculate_density_and_counts(image_path):
    # Predict with the YOLO model
    results = model.predict(source=image_path, conf=0.4)
    image = cv2.imread(image_path)
    total_density = 0
    vehicle_counts = {key: 0 for key in vehicle_area.keys()}

    for bbox in results[0].boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        class_id = int(bbox.cls[0])
        class_name = results[0].names[class_id]

        # Draw bounding boxes and labels
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (36, 255, 24), 3)

        # Calculate density based on vehicle class
        if class_name in vehicle_area:
            total_density += vehicle_area[class_name]
            vehicle_counts[class_name] += 1  # Increment the count for this class

    # Save the processed image
    processed_image_path = os.path.join('uploads', 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, image)

    return total_density, processed_image_path, vehicle_counts

@app.route('/predict', methods=['POST'])
def predict():
    lanes = ['lane1', 'lane2', 'lane3', 'lane4']
    densities = {}
    processed_image_paths = {}
    vehicle_counts = {}

    # Loop over the four lane images
    for lane in lanes:
        if lane not in request.files:
            return {'error': f'{lane} image file not provided'}, 400

        image_file = request.files[lane]
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)

        # Calculate density and vehicle counts for each lane
        density, processed_image_path, counts = calculate_density_and_counts(image_path)
        densities[lane] = density
        processed_image_paths[lane] = processed_image_path
        vehicle_counts[lane] = counts  # Store counts for each lane

        # Remove the original image after processing
        os.remove(image_path)

    # Return the images, densities, and vehicle counts to the frontend
    return jsonify({
        'densities': densities,
        'image_urls': {lane: request.host_url + 'uploads/processed_' + os.path.basename(processed_image_paths[lane]) for lane in lanes},
        'vehicle_counts': vehicle_counts  # Return vehicle counts
    })

@app.route('/uploads/<filename>')
def get_image(filename):
    return send_file(os.path.join('uploads', filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
