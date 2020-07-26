import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("../../datasets/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        # print(landmarks.parts())
        nose = landmarks.parts()[27]
        # print(nose.x, nose.y)
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (255, 0, 0), 3)

    # print(faces)

    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()