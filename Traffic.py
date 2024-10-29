import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('traffic.pt')  # You can replace 'best.pt' with your own trained model path

# Class names for traffic signs
classNames = ["bus_stop", "do_not_enter", "do_not_stop", "do_not_turn_l", "do_not_turn_r", "do_not_u_turn",
              "enter_left_lane", "green_light", "left_right_lane", "no_parking", "parking", "ped_crossing",
              "ped_zebra_cross", "railway_crossing", "red_light", "stop", "t_intersection_l", "traffic_light",
              "u_turn", "warning", "yellow_light"]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break
    
    # Perform object detection using YOLOv8
    results = model.predict(source=img, save=False, show=False)

    # Extract prediction results: bounding boxes, class labels, and confidence scores
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()   # Bounding boxes
        labels = result.boxes.cls.cpu().numpy()   # List of class indices (labels)
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores

        # Loop over each detection and display the bounding box, label, and confidence score
        for box, cls, score in zip(boxes, labels, scores):
            # Get the corresponding traffic sign for each detected label
            traffic_sign = classNames[int(cls)]
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box)
            
            # Display the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display the label and confidence score above the bounding box
            label_text = f"{traffic_sign}: {score:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display the result frame
    cv2.imshow("Webcam Detection", img)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
