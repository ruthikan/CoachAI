import pandas as pd
import joblib
import google.generativeai as genai

# Load Diet model
rf_diet = joblib.load('rf_diet.pkl')
le_diet = joblib.load('le_diet.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# Configure Gemini
genai.configure(api_key="AIzaSyBAXeFiPCsxDpO2chKPk7v_5Ye4EYItjJo")

def recommend_diet_plan(name, age, height, weight, gender, health_condition, diet_preference):
    # Prepare input
    input_data = {
        'Age': [age],
        'Height_cm': [height],
        'Weight_kg': [weight],
    }

    for g in ["Male", "Female"]:
        input_data[f"Gender_{g}"] = [1 if gender == g else 0]

    for h in ["None", "Diabetes", "Hypertension"]:
        input_data[f"HealthCondition_{h}"] = [1 if health_condition == h else 0]

    for d in ["Veg", "Non-Veg"]:
        input_data[f"DietPreference_{d}"] = [1 if diet_preference == d else 0]

    # Align with training features
    new_user = pd.DataFrame(input_data)
    for col in feature_cols:
        if col not in new_user.columns:
            new_user[col] = 0
    new_user = new_user[feature_cols]

    # Prediction
    predicted_diet = le_diet.inverse_transform(rf_diet.predict(new_user))[0]

    # Generate 30-day meal plan
    prompt = f"""
    🍽️ Create a **30-day meal plan** for:
    👤 {name}, {age} yrs, {height} cm, {weight} kg, {gender}
    ❤️ Health condition: {health_condition}
    🥗 Diet Preference: {diet_preference}
    📋 Recommended Diet Type: {predicted_diet}
    📅 Each day should include Breakfast, Lunch, Snacks & Dinner.
    👉 Add emojis for attractiveness.
    """

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)

    return predicted_diet, response.text


if __name__ == "__main__":
    print("🥗 Diet Recommendation System 🥗")
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    height = int(input("Enter your height (cm): "))
    weight = int(input("Enter your weight (kg): "))
    gender = input("Enter your gender (Male/Female): ")
    health_condition = input("Enter health condition (None/Diabetes/Hypertension): ")
    diet_preference = input("Do you prefer Veg or Non-Veg diet? ")

    diet, plan = recommend_diet_plan(name, age, height, weight, gender, health_condition, diet_preference)

    print(f"\n✅ {name}, your recommended diet type is: {diet}")
    print("\n📅 Your 30-Day Diet Plan:\n")
    print(plan)
