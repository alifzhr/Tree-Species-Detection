import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import time

model = YOLO("best.pt")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (320,320)}))
picam2.start()

prev_time = time.time()

while True:
    frame = picam2.capture_array()
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    results = model(frame_bgr, conf = 0.15, verbose = False)
    
    annotated_frame = results[0].plot()
    
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        2,
    )
    
    cv2.imshow("Tree Detection - LIVE", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cv2.destroyAllWindows()
picam2.stop()