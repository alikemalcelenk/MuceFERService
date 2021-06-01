from flask import Flask

# ---------------------------------
# API
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Muce - Facial Emotion Recognition API Service'


if __name__ == "__main__":
    app.run()
