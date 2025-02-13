# Bitcoin Price Prediction Web App

# 📌 Overview

This project predicts Bitcoin's future closing prices using a deep learning model. The model is trained on historical BTC price data and deployed as a Flask web application, allowing users to input a stock ticker and predict future prices with interactive visualizations.

# 📂 Project Structure

├── data/            # Raw and preprocessed datasets  
├── model/           # Code for model training & saved model  
├── api/             # Flask API for predictions  
├── templates/       # HTML templates for web app  
├── static/          # Static files (CSS, JS)  
├── notebooks/       # Jupyter Notebooks for analysis  
├── app.py           # Main Flask application  
├── model.keras      # Pre-trained model file  
├── requirements.txt # Required dependencies  
└── README.md        # Project documentation  

# 🛠 Technologies Used

1. Python (pandas, numpy, scikit-learn, yfinance)
2. Deep Learning (LSTM, TensorFlow/Keras)
3. Flask (for API & Web UI)
4. Matplotlib (for visualizations)

# 🚀 How to Run the Project

## 1️⃣ Clone the Repository

git clone https://github.com/your-username/btc-price-prediction.git

cd btc-price-prediction

## 2️⃣ Install Dependencies

pip install -r requirements.txt

## 3️⃣ Run the Flask App

python app.py
The app will run at http://127.0.0.1:5000/.

# 📊 API Usage

1️⃣ Prediction Request

Endpoint:POST /predict
Input: JSON with required features
Output: Predicted BTC closing price

Example Request:

{
  "open": 45000,
  "high": 46000,
  "low": 44500,
  "volume": 15000
}

Example Response:

{
  "predicted_close": 45500.32
}

# 📈 Features

✅ Fetches live BTC data from Yahoo Finance

✅ Pre-trained LSTM model for accurate predictions

✅ Flask web UI for easy interaction

✅ Visualizations for actual vs predicted prices

# 🐝 Evaluation Criteria

1. Data Preprocessing: Handling missing values, feature scaling

2. Model Performance: RMSE, MAE evaluation

3. API Functionality: Proper request handling & response

4. Bonus: Deployment & UI enhancements

# 👨‍💻 Contributing

Feel free to fork, open issues, or submit pull requests!

📧 Contact: [yashasvibhardwajwork@gmail.com / yashasvi1110]
