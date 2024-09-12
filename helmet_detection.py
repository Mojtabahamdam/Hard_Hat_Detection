import cv2
import numpy as np
import torch

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 480))  # Resize to the expected input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # Convert to PyTorch tensor and permute dimensions
    frame = torch.div(frame, 255.0)  # Normalize pixel values to [0, 1]
    return frame.unsqueeze(0)  # Add batch dimension

def detect_objects_in_video(video_source, output_path, conf_thres=0.5):
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
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        
        # Perform inference
        results = model(preprocessed_frame)
        
        # Render results on the frame
        for result in results.pred[0]:  # Iterate over detected objects
            x1, y1, x2, y2, conf, cls = result.tolist()  # Convert tensor to list
            class_name = model.names[int(cls)]
            
            if class_name == 'hard_hat':
                color = (255, 0, 0)  # Color for hard hats
                # Draw the bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif class_name == 'head':
                color = (0, 255, 0)  # Color for heads
                # Draw the bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                continue  # Skip if not a head or hard hat

        # Write the frame with detections to the output video
        out.write(frame)
        
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
detect_objects_in_video(video_source, output_path, conf_thres=0.5)