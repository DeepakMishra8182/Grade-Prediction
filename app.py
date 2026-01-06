from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("student_pass_fail_10marks.pkl")

health_map = {
    "very bad": 1,
    "bad": 2,
    "average": 3,
    "good": 4,
    "excellent": 5
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        G1 = float(request.form['G1'])
        G2 = float(request.form['G2'])
        studytime = float(request.form['studytime'])
        absences = float(request.form['absences'])
        failures = 1 if request.form['failures'] == "yes" else 0
        schoolsup = 1 if request.form['schoolsup'] == "yes" else 0
        famsup = 1 if request.form['famsup'] == "yes" else 0
        internet = 1 if request.form['internet'] == "yes" else 0
        health = health_map[request.form['health']]
        if not (0 <= G1 <= 10 and 0 <= G2 <= 10):
            return render_template("index.html",
                prediction_text="❌ G1 & G2 must be between 0 and 10")

        internal_avg = (G1 + G2) / 2
        attendance_score = max(0, 10 - absences / 5)

        features = np.array([[
            G1, G2, studytime, failures, absences,
            schoolsup, famsup, internet, health,
            internal_avg, attendance_score
        ]])

        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template(
            "index.html",
            prediction_text=f"🎯 Predicted Final Score (G3): {prediction} / 10"
        )

    except Exception as e:
        return render_template("index.html",
            prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
