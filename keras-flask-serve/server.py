from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

app = flask.Flask(__name__)
model = None

# Preprocess the input image from the user
def preprocess_image(image, input_shape):
  image = image.convert('L')
  image = image.resize(input_shape)
  image = img_to_array(image)
  image = image.reshape(-1, 28, 28, 1)
  image /= 255
  return image

# Route for /predict
@app.route("/predict", methods=["POST"])
def predict():
    result = {"success": False}

    if flask.request.method == "POST":
      if flask.request.files.get("input"):
        image = flask.request.files["input"].read()
        image = Image.open(io.BytesIO(image))
        image = preprocess_image(image, input_shape=(28, 28))

        result["predictions"] = []

        for (label, prob) in enumerate(preds[0]):
          r = {"label": label, "probability": float(prob)}
          result["predictions"].append(r)

        result["success"] = True

    return flask.jsonify(result)

if __name__ == "__main__":
  print("* Loading Keras model and Flask starting server...")
  model = load_model('test_model/test_model.h5')
  app.run(port=8080)