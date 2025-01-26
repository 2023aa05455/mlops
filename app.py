from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__,template_folder='templates')

# Load the pre-trained model and scaler
model_path = "app/lr_model.joblib"
scaler_path = "app/scaler.joblib"
# Ensure that the model and scaler files exist before loading
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully!")
else:
    print("Error: Model or scaler file not found!")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    # return jsonify({"message": "Breast Cancer Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from POST request
        input_data = request.json.get("data")
        if not input_data:
            return jsonify({"error": "No input data provided!"}), 400

        # Convert input data to numpy array and scale it
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": prediction_proba[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
