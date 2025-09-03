from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("eligibility_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [int(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)

    # Predict eligibility
    prediction = model.predict(features_array)[0]

    # Build result text
    if prediction == 1:
        prediction_text = "✅ Eligible for External Exam"
        suggestions = ["Keep up the good work!", "Maintain your consistency."]
    else:
        prediction_text = "❌ Not Eligible for External Exam"
        suggestions = [
            "Increase your attendance 📅",
            "Focus more on lab performance 🧪",
            "Participate in class activities 🎤",
            "Be more disciplined ⏰"
        ]

    # Fake class average (for radar chart)
    class_avg = [70, 75, 65, 80, 60, 85]

    chart_data = {
        "student": features,   # Student entered values
        "class_avg": class_avg # Class average
    }

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        suggestions=suggestions,
        chart_data=chart_data
    )

if __name__ == "__main__":
    app.run(debug=True)
