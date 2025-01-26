import pickle
from flask import Flask, request, jsonify

# Load the trained model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get input JSON
        features = data["features"]  # Extract features from JSON
        prediction = model.predict([features])  # Make prediction
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
