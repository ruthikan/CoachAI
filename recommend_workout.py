import pandas as pd
import joblib
import google.generativeai as genai

# Load Workout model
rf_workout = joblib.load('rf_workout.pkl')
le_workout = joblib.load('le_workout.pkl')
feature_cols = joblib.load('feature_columns.pkl')

# Configure Gemini
genai.configure(api_key="AIzaSyBAXeFiPCsxDpO2chKPk7v_5Ye4EYItjJo")

def recommend_workout_plan(name, age, height, weight, gender, level, goal, workouts_per_week, walks_per_week, health_condition):
    # Prepare input
    input_data = {
        'Age': [age],
        'Height_cm': [height],
        'Weight_kg': [weight],
        'WorkoutsPerWeek': [workouts_per_week],
        'WalksPerWeek': [walks_per_week],
    }

    # One-hot categorical encoding
    for g in ["Male", "Female"]:
        input_data[f"Gender_{g}"] = [1 if gender == g else 0]

    for l in ["Beginner", "Intermediate", "Professional"]:
        input_data[f"Level_{l}"] = [1 if level == l else 0]

    for g in ["Endurance", "Flexibility", "Muscle Gain", "Weight Loss"]:
        input_data[f"FitnessGoal_{g}"] = [1 if goal == g else 0]

    for h in ["None", "Diabetes", "Hypertension"]:
        input_data[f"HealthCondition_{h}"] = [1 if health_condition == h else 0]

    # Align with training features
    new_user = pd.DataFrame(input_data)
    for col in feature_cols:
        if col not in new_user.columns:
            new_user[col] = 0
    new_user = new_user[feature_cols]

    # Prediction
    predicted_workout = le_workout.inverse_transform(rf_workout.predict(new_user))[0]

    # Generate 30-day workout plan
    prompt = f"""
    🎯 Create a **30-day progressive workout plan** for:
    👤 {name}, {age} yrs, {height} cm, {weight} kg, {gender}
    🥅 Goal: {goal}
    💪 Recommended Workout Type: {predicted_workout}
    📅 Each day should include exercises with reps/sets.
    👉 Add emojis for motivation & variety.
    """

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)

    return predicted_workout, response.text


if __name__ == "__main__":
    print("🏋️ Workout Recommendation System 🏋️")
    name = input("Enter your name: ")
    age = int(input("Enter your age: "))
    height = int(input("Enter your height (cm): "))
    weight = int(input("Enter your weight (kg): "))
    gender = input("Enter your gender (Male/Female): ")
    level = input("Enter your level (Beginner/Intermediate/Professional): ")
    goal = input("Enter your fitness goal (Endurance/Flexibility/Muscle Gain/Weight Loss): ")
    workouts_per_week = int(input("How many days you workout per week? "))
    walks_per_week = int(input("How many days you walk per week? "))
    health_condition = input("Enter health condition (None/Diabetes/Hypertension): ")

    workout, plan = recommend_workout_plan(name, age, height, weight, gender, level, goal, workouts_per_week, walks_per_week, health_condition)

    print(f"\n✅ {name}, your recommended workout type is: {workout}")
    print("\n📅 Your 30-Day Workout Plan:\n")
    print(plan)
