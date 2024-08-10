from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model_path = r"C:\Users\User\Documents\NDSIC\Deploy\House\gradient_boosting_model.pkl"
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    # Predict using the loaded model
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text=f'Predicted Value: ${prediction[0]:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
