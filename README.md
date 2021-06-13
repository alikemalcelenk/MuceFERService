# MuceFERService

Muce is my graduation project that suggests music to users by predicting facial emotions using CNN from their images. Emotions that are predicted are happiness, neutral, sadness and anger. This repository is code base of facial emotion recognition API service. It is written in Python using the Flask and Keras frameworks.

## Link

- [🌍 &nbsp; Muce API Base URL](https://muce-fer-api-service.herokuapp.com/)

## Documentation

| Route   | HTTP Verb | POST body             | Description                 |
| ------- | --------- | --------------------- | --------------------------- |
| /upload | `POST`    | {'image': test.jpeg } | Predict emotion from image. |

## Tech Stack

- flask
- tensorflow
- keras
- opencv-python
- numpy
- sklearn
- dlib
- gunicorn
