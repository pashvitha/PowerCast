# PowerCast
Developed a machine learning system using Random Forest and LightGBM to predict power consumption with weather and time features, integrated with a Tkinter GUI for real-time forecasts.
1)Install dependencies
Run the following command once:
pip install pandas scikit-learn lightgbm joblib tk

2)Train the LightGBM model
Run:
python trainmodel.py

3)Run the GUI with LightGBM
python power_gui.py

4)Use the application
Enter Temperature, Humidity, Wind Speed, General Diffuse Flows, Diffuse Flows.
Click Predict.
The app will display Power Consumption (W) and in some versions Energy (kWh) and Estimated Cost.
