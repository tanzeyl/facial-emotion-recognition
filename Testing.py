import cv2
import dlib
import time
import joblib
import argparse
import numpy as np
from imutils.face_utils import rect_to_bb
from tensorflow.keras.models import load_model
import utils

hog_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("Face Detector/shape_predictor_68_face_landmarks.dat")

def get_model_compatible_input(gray_frame, face):
        img_arr = utils.align_face(gray_frame, face, desiredLeftEye)
        img_arr = utils.preprocess_img(img_arr, resize=False)
        landmarks = shape_predictor(gray_frame, face,)
        roi1, roi2 = utils.extract_roi1_roi2(gray_frame, landmarks)
        roi1 = np.expand_dims(roi1, 0)
        roi2 = np.expand_dims(roi2, 0)
        roi1 = roi1 / 255.
        roi2 = roi2 / 255.
        return [img_arr, roi1, roi2]

def dlib_detector(frame):
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    offset = 15
    x_pos,y_pos = 10,40
    faces = hog_detector(gray_frame)
    for idx, face in enumerate(faces):
        model_in = get_model_compatible_input(gray_frame, face)
        predicted_proba = model.predict(model_in)
        predicted_label = np.argmax(predicted_proba[0])
        x,y,w,h = rect_to_bb(face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        text = f"Person {idx+1}: {label2text[predicted_label]}"
        utils.draw_text_with_backgroud(frame, text, x + 5, y, font_scale=0.4)
        text = f"Person {idx+1} :  "
        y_pos = y_pos + 2*offset
        utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))
        for k,v in label2text.items():
            text = f"{v}: {round(predicted_proba[0][k]*100, 3)}%"
            y_pos = y_pos + offset
            utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))


desiredLeftEye=(0.32, 0.32)
model = load_model("best_model.h5")
label2text = joblib.load("label2text_CNNModel_ck_5emo.pkl")


if __name__ == "__main__":
    vidcap=cv2.VideoCapture(0)
    frame_count = 0
    tt = 0
    while True:
        status, frame = vidcap.read()
        if not status:
            break
        frame_count += 1

        try:
            tik = time.time()
            out = dlib_detector(frame)
            tt += time.time() - tik
            fps = frame_count / tt
        except Exception as e:
            print(e)
            pass

        cv2.imshow("Face Detection Comparison", frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()
    vidcap.release()
