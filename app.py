from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialize Flask
app = Flask(__name__)

# Load model and scaler
model = joblib.load("mental_health_model.pkl")
scaler_age = joblib.load("scaler_age.pkl")

# Feature columns
feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        age = float(request.form['age'])
        gender = request.form['gender']
        family_history = request.form['family_history']
        benefits = request.form['benefits']
        care_options = request.form['care_options']
        anonymity = request.form['anonymity']
        leave = request.form['leave']
        work_interfere = request.form['work_interfere']

        # Encode categorical features
        gender = 1 if gender.lower() == 'male' else 0
        family_history = 1 if family_history.lower() == 'yes' else 0
        benefits = 1 if benefits.lower() == 'yes' else 0
        care_options = 1 if care_options.lower() == 'yes' else 0
        anonymity = 1 if anonymity.lower() == 'yes' else 0

        leave_map = {'very difficult': 0, 'difficult': 1, 'somewhat easy': 2, 'easy': 3}
        leave = leave_map.get(leave.lower(), 1)

        work_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3}
        work_interfere = work_map.get(work_interfere.lower(), 2)

        # ✅ Scale only Age
        age_scaled = scaler_age.transform([[age]])[0][0]

        # Create final input array
        input_data = np.array([[age_scaled, gender, family_history, benefits,
                                care_options, anonymity, leave, work_interfere]])

        # Predict
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][prediction] * 100

        # Interpret result
        if prediction == 1:
            result = f"🧠 The person is likely to need mental health treatment ({prob:.1f}% confidence)."
        else:
            result = f"😊 The person is unlikely to need mental health treatment ({prob:.1f}% confidence)."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
