from flask import Flask, request
import os
import cv2
import dlib
import numpy as np

from keras.models import load_model
import os


# ---------------------------------
# LOADING MODEL

loadedModal = load_model('main.model')

# ---------------------------------
# CROPPING IMAGE FUNCTIONS

detector = dlib.get_frontal_face_detector()
new_path = './crops/'


def MyRec(rgb, x, y, w, h, v=20, color=(200, 0, 0), thikness=2):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x, y), (x+v, y), color, thikness)
    cv2.line(rgb, (x, y), (x, y+v), color, thikness)

    cv2.line(rgb, (x+w, y), (x+w-v, y), color, thikness)
    cv2.line(rgb, (x+w, y), (x+w, y+v), color, thikness)

    cv2.line(rgb, (x, y+h), (x, y+h-v), color, thikness)
    cv2.line(rgb, (x, y+h), (x+v, y+h), color, thikness)

    cv2.line(rgb, (x+w, y+h), (x+w, y+h-v), color, thikness)
    cv2.line(rgb, (x+w, y+h), (x+w-v, y+h), color, thikness)


def save(img, name, bbox, width=48, height=48):
    x, y, w, h = bbox
    imgCrop = img[y:h, x: w]
    # we need this line to reshape the images
    imgCrop = cv2.resize(imgCrop, (width, height))
    cv2.imwrite(name, imgCrop)


def faces(image, imageFilename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    # detect the face
    for counter, face in enumerate(faces):
        print(counter)
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (220, 255, 220), 1)
        MyRec(image, x1, y1, x2 - x1, y2 - y1, 10, (0, 250, 0), 3)
        save(gray, "./images/" + imageFilename, (x1, y1, x2, y2))
    newImg = cv2.imread("./images/" + imageFilename)
    print("done saving")
    return newImg


# ---------------------------------
# API
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Muce - Facial Emotion Recognition Service'


@app.route('/upload', methods=['POST'])
def upload():
    try:
        image = request.files['image']
        if not image:
            return {'message': 'No image uploaded!'}, 404

        image.save(os.path.join("images/", image.filename))
        inputImg = cv2.imread("./images/" + image.filename)

        cropedImg = faces(inputImg, image.filename)
        cropedImg = np.array([cropedImg])  # to prepare to tensorflow
        cropedImg = cropedImg/255  # normalization

        predClass = np.argmax(loadedModal.predict(cropedImg), axis=-1)
        predNumber = predClass[0]

        if (predNumber == 0):
            return {'index': '0', 'emotion': 'happy'}
        elif (predNumber == 1):
            return {'index': '1', 'emotion': 'fear'}
        elif (predNumber == 2):
            return {'index': '2', 'emotion': 'surprise'}
        elif (predNumber == 3):
            return {'index': '3', 'emotion': 'sadness'}
        elif (predNumber == 4):
            return {'index': '4', 'emotion': 'neutral'}
        elif (predNumber == 5):
            return {'index': '5', 'emotion': 'anger'}
        elif (predNumber == 6):
            return {'index': '6', 'emotion': 'disgust'}
        else:
            return {'message': 'There is not exist any true prediction value!'}, 404
    except:
        return {'message': 'There is an error occurred! Please try again.'}, 500


if __name__ == "__main__":
    app.run()
