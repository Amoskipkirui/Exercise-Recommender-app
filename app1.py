from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained Random Forest model
with open('rf4_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Define a dictionary of intensity levels and descriptions
intensity_descriptions = {
    1: 'Very low intensity: Suitable for individuals with minimal physical activity, such as walking at a leisurely pace, stretching or yoga, and gentle swimming. However, please note that your preference is prioritized.',
    2: 'Low intensity: Suitable for beginners or individuals with limited physical activity, such as light jogging or running, cycling at a relaxed pace, and beginner\'s aerobics. However, please remember that your preference takes priority.',
    3: 'Moderate intensity: Suitable for individuals who engage in regular physical activity, including brisk walking, dancing, and water aerobics. However, please keep in mind that your preference is given priority.',
    4: 'Medium intensity: Suitable for individuals with moderate fitness and physical activity, such as power walking, cycling at a moderate pace, and Zumba. However, please remember that your preference is given priority.',
    5: 'Moderate to high intensity: Suitable for individuals with moderate to high fitness levels, including jogging or running at a moderate pace, high-intensity interval training (HIIT), and kickboxing. However, please note that your preference is given priority.',
    6: 'High intensity: Suitable for individuals with a high level of fitness and physical activity, such as running at a fast pace, circuit training, and CrossFit. However, please keep in mind that your preference is given priority.',
    7: 'High intensity: Suitable for individuals with a high level of fitness and physical activity, such as advanced HIIT workouts, competitive sports (e.g., soccer, basketball), and spinning or indoor cycling classes. However, please remember that your preference is given priority.',
    8: 'Very high intensity: Suitable for individuals with very high fitness and physical activity, including sprinting or interval sprints, plyometric exercises, and heavy weightlifting. However, please keep in mind that your preference is given priority.',
    9: 'Very higher intensity: Suitable for individuals with very high fitness and physical activity, such as advanced CrossFit workouts, box jumps, and Olympic weightlifting. However, please note that your preference is given priority.',
    10: 'Extremely high intensity: Suitable for athletes or individuals with exceptional fitness levels, including professional sports training, marathon running, and elite-level strength and conditioning programs. However, please remember that your preference is given priority.'
}

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Retrieve the form data
    calories_to_burned = float(request.form['calories'])
    dream_weight = float(request.form['dream_weight'])
    actual_weight = float(request.form['actual_weight'])
    age = float(request.form['age'])
    duration = float(request.form['duration'])
    height = float(request.form['height'])
    weather_conditions = request.form['weather_conditions']
    gender = request.form['gender']

    # Create a DataFrame with user input
    user_input = pd.DataFrame(
        [[calories_to_burned, dream_weight, actual_weight, age, duration, height, weather_conditions, gender]],
        columns=['Calories to Burn (Kcal)', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Height', 'Weather Conditions', 'Gender'])

    # One-hot encode categorical variables
    user_input_encoded = pd.get_dummies(user_input, columns=['Weather Conditions', 'Gender'])

    # Get the column names of the training data
    training_columns = pd.get_dummies(pd.DataFrame(columns=user_input_encoded.columns)).columns

    # Reindex user input DataFrame to match the training data columns
    user_input_encoded = user_input_encoded.reindex(columns=training_columns, fill_value=0)

    # Check if user_input_encoded has at least one row
    if user_input_encoded.empty:
        # Handle the empty DataFrame case
        return render_template('error.html', message='Invalid input data')

    # Make predictions on the user input
    predicted_intensity_selected = int(round(rf_model.predict(user_input_encoded.values)[0]))

    # Retrieve the description based on the predicted intensity level
    description = intensity_descriptions.get(predicted_intensity_selected, "Unknown")

    # Pass the results to the recommendation page
    return render_template('recommendation.html', intensity=predicted_intensity_selected, description=description)

if __name__ == '__main__':
    app.run(debug=True)
