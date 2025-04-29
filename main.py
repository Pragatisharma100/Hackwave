from flask import Flask, request, render_template, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai

# Configure Gemini API (only once)
GEMINI_API_KEY = "AIzaSyBiMebMFBzLUjXrKFth7zHEtp29HfLgLng"  # Your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro")

app = Flask(__name__)
app.secret_key = 'secretkey'

# Load CSV data
sym_des = pd.read_csv("symptoms.csv")
precautions = pd.read_csv("precautions.csv")
workout = pd.read_csv("workout.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")

# Load model
svc = pickle.load(open('svc.pkl', 'rb'))

# Extract symptoms and map
symptoms_list = sym_des['Symptom'].str.strip().str.lower().tolist()
valid_symptoms = set(symptoms_list)
symptoms_dict = {symptom: idx for idx, symptom in enumerate(symptoms_list)}

# Disease list as per model output
diseases_list = {
    0: '(vertigo) Paroymsal Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis',
    4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis',
    8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue',
    12: 'Diabetes ', 13: 'Dimorphic hemmorhoids(piles)', 14: 'Drug Reaction',
    15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    23: 'Hypertension ', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism',
    27: 'Impetigo', 28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthristis',
    32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer diseae', 34: 'Pneumonia',
    35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection',
    39: 'Varicose veins', 40: 'hepatitis A'
}

# Helper to extract details
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc) if not desc.empty else "Description not available."

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [list(row) for row in pre.values] if not pre.empty else [["Precautions not available."]]

    med = medications[medications['Disease'] == dis]['Medication']
    med = med.tolist() if not med.empty else ["Medications not available."]

    die = diets[diets['Disease'] == dis]['Diet']
    die = die.tolist() if not die.empty else ["Diet not available."]

    workout_plan = workout[workout['disease'] == dis]['workout']
    workout_plan = workout_plan.tolist() if not workout_plan.empty else ["Workout plan not available."]

    return desc, pre, med, die, workout_plan

# Generate explanation from Gemini
def get_gemini_explanation(disease_name, symptoms):
    prompt = f"I have the following symptoms: {', '.join(symptoms)}. The system predicted '{disease_name}'. Please explain why and how these are related."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return "AI explanation not available due to an error."
    
# Prediction function
def get_predicted_value(patient_symptoms):
    invalid_symptoms = [sym for sym in patient_symptoms if sym.lower() not in valid_symptoms]
    if invalid_symptoms:
        return None, invalid_symptoms

    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    predicted_disease = svc.predict([input_vector])[0]
    return diseases_list.get(predicted_disease, "Unknown Disease"), None

# Routes
@app.route('/', methods=['GET', 'POST'])  # Allow both GET and POST
def index():
    result = ""
    if request.method == "POST":
        user_input = request.form["prompt"]
        try:
            response = model.generate_content(user_input)
            result = response.text
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template("index.html", result=result, symptoms=symptoms_list)


@app.route('/predict', methods=['POST'])
def predict():
    user_symptoms = request.form.getlist('symptoms')

    if not user_symptoms:
        flash("Please select symptoms!")
        return redirect(url_for('index'))

    predicted_disease, invalid_symptoms = get_predicted_value(user_symptoms)

    if invalid_symptoms:
        error_msg = f"Invalid symptom(s): {', '.join(invalid_symptoms)}. Please enter correct spelling."
        return render_template('index.html', error_msg=error_msg, symptoms=symptoms_list)

    desc, pre, med, die, workout = helper(predicted_disease)
    gemini_summary = get_gemini_explanation(predicted_disease, user_symptoms)

    return render_template(
        'index.html',
        predicted_disease=predicted_disease,
        dis_des=desc,
        dis_pre=pre,
        dis_med=med,
        dis_die=die,
        dis_workout=workout,
        gemini_explanation=gemini_summary,
        symptoms=symptoms_list
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contacts():
    return render_template('contact.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

if __name__ == "__main__":
    app.run(debug=True)
