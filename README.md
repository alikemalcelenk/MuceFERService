# MuceFERService

Muce is my graduation project that suggests music to users by predicting facial emotions from their images. This repository is code base of facial emotion recognition API service written in Python(Flask) with Keras.

## Link

- [🌍 &nbsp; Muce API Base URL](https://muce-fer-api-service.herokuapp.com/)

## Documentation

| Route   | HTTP Verb | POST body             | Description                 |
| ------- | --------- | --------------------- | --------------------------- |
| /upload | `POST`    | {'image': test.jpeg } | Predict emotion from image. |

## Tech Stack

- tensorflow
- keras
- flask
- opencv-python
- numpy
- sklearn
- dlib
- gunicorn
