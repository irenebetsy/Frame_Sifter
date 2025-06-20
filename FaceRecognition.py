import cv2
import mediapipe as mp

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Open video file
cap = cv2.VideoCapture('video_path')

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face Detection
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                cv2.rectangle(frame, bbox, (255, 0, 0), 2)

                # Face Mesh
                mesh_results = face_mesh.process(rgb_frame)
                if mesh_results.multi_face_landmarks:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        for landmark in face_landmarks.landmark:
                            # Process landmarks here for features like eyes, mustache, etc.
                            pass

        cv2.imshow('Face Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
