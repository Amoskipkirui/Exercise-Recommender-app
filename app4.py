from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

import pickle
# Load the pickled model
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

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

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the user input from the form
        calories_to_burned = float(request.form['calories'])
        dream_weight = float(request.form['dream_weight'])
        actual_weight = float(request.form['actual_weight'])
        age = float(request.form['age'])
        duration = float(request.form['duration'])
        height = float(request.form['height'])
        weather_conditions = request.form['weather_conditions']
        gender = request.form['gender']

        # Calculate the maximum heart rate
        max_heart_rate = 220 - age

        # Create a DataFrame with the user input
        user_input = pd.DataFrame(
            [[calories_to_burned, dream_weight, actual_weight, age, duration, height, max_heart_rate, weather_conditions, gender]],
            columns=['Calories to Burn (Kcal)', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Height', 'maximum Heart Rate', 'Weather Conditions', 'Gender']
        )

        # Make predictions on the user input
        predicted_intensity = int(round(loaded_model.predict(user_input)[0]))
        description = intensity_descriptions.get(predicted_intensity, "Unknown")

        return render_template('result.html', intensity=predicted_intensity, description=description)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
