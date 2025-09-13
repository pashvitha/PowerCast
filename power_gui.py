
import tkinter as tk
from tkinter import messagebox
import joblib
import datetime 

# Load the model
model = joblib.load("power_consumption_lgbm.pkl")

def predict():
    try:
        # Get inputs from GUI
        inputs = [
            float(temp_entry.get()),
            float(humidity_entry.get()),
            float(wind_entry.get()),
            float(gdf_entry.get()),
            float(df_entry.get())
        ]

        # Automatically add current time-based features
        now = datetime.datetime.now()
        inputs.extend([
            now.hour,
            now.day,
            now.month,
            now.weekday(),
            int(now.weekday() >= 5)  # is_weekend
        ])

        # Make prediction
        prediction = model.predict([inputs])[0]
        result_label.config(text=f"Predicted Power Consumption: {prediction:.2f} W")
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numbers.")


# Tkinter UI
root = tk.Tk()
root.title("Power Consumption Predictor")

tk.Label(root, text="Temperature").grid(row=0, column=0)
temp_entry = tk.Entry(root)
temp_entry.grid(row=0, column=1)

tk.Label(root, text="Humidity").grid(row=1, column=0)
humidity_entry = tk.Entry(root)
humidity_entry.grid(row=1, column=1)

tk.Label(root, text="Wind Speed").grid(row=2, column=0)
wind_entry = tk.Entry(root)
wind_entry.grid(row=2, column=1)

tk.Label(root, text="General Diffuse Flows").grid(row=3, column=0)
gdf_entry = tk.Entry(root)
gdf_entry.grid(row=3, column=1)

tk.Label(root, text="Diffuse Flows").grid(row=4, column=0)
df_entry = tk.Entry(root)
df_entry.grid(row=4, column=1)

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=5, columnspan=2, pady=10)

result_label = tk.Label(root, text="Prediction will appear here.")
result_label.grid(row=6, columnspan=2)

root.mainloop()
