import cv2
import numpy as np
import torch

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def detect_objects_in_video(video_source, output_path):
    # Open the video source (can be a URL or a local file)
    cap = cv2.VideoCapture(video_source)
    
    # Check if the video source opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 FPS if not available
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'mp4v' depending on your preference
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Perform inference
        results = model(frame)
        
        # Render results on the frame
        for result in results.xyxy[0]:  # Iterate over detected objects
            # Extract bounding box coordinates and class ID
            x1, y1, x2, y2, conf, cls = result
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Define colors for different classes
            if int(cls) == 1:  # Assuming class ID 1 corresponds to helmets
                color = (255, 0, 0)  # Green for helmets
            elif int(cls) == 0:  # Assuming class ID 0 corresponds to heads
                color = (0, 255, 0)  # Red for heads
            else:
                continue  # Skip if not a head or helmet
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Optionally, put a label with the confidence score
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the frame with detections to the output video
        out.write(frame)  # No need to convert color here since OpenCV uses BGR format
        
        # Display the frame with detections
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to {output_path}")

# Example usage for webcam:
video_source = 0  # Use 0 for the default webcam
output_path = 'output_video.mp4'
detect_objects_in_video(video_source, output_path)