# AP_states_cropyield_prediction
Crop Yield Prediction (tonnes/hectare) for 12 Unique Crops in Andhra Pradesh Districts

Here's a **well-structured README.md** file for your project:  

```markdown
# Crop Yield Prediction in Andhra Pradesh 🌾  

## Overview  
This project predicts **crop yield (tonnes/hectare) for 12 unique crops** across **13 districts** of Andhra Pradesh using **machine learning models**. The dataset covers **2013-2022** and has undergone extensive preprocessing in a separate repository.  

## Features  
- **Enhanced Decision Tree & Random Forest Regressors** for improved accuracy.  
- **Flask-based UI** (`ap.py`) for user interaction.  
- **Comprehensive data preprocessing** (handled in a separate repository).  
- **Visualization of results** via generated graphs.  

## Dataset  
- Data collected from **2013-2022** across Andhra Pradesh districts.  
- Features include **Crop Type, District, Season, Area, Annual Rainfall, Fertilizer Usage, and Temperature**.  
- Preprocessed dataset available in [ML_DataPreProcessing Repository](https://github.com/Anudeep007-hub/ML_DataPreProcessing.git).  

## Project Structure  
```
├── ap.py                   # Flask-based user interface for yield prediction  
├── Model3.py               # ML models (Enhanced Decision Tree & Random Forest)  
├── final_data.csv          # Preprocessed dataset  
├── enhanced_random_forest_regressor.pkl  # Trained ML model  
├── label_encoders.pkl      # Encoded categorical variables  
├── test.ipynb              # Exploratory data analysis & model evaluation  
├── static/                 # Static files for Flask app  
├── templates/              # HTML templates for UI  
└── README.md               # Project Documentation  
```

## Installation & Setup  
### Prerequisites  
Ensure you have **Python 3.8+** and install dependencies using:  
```bash
pip install -r requirements.txt
```

### Running the Project  
1. **Train the Model (if needed)**  
   ```bash
   python main.py
   ```
2. **Run the Flask App**  
   ```bash
   python ap.py
   ```
3. Open **http://127.0.0.1:5000/** in your browser to access the UI.  

## Usage  
- **Train a new model**: Option to train a fresh ML model.  
- **Evaluate model performance**: Get metrics like MAE, RMSE, and R² Score.  
- **Make Predictions**: Enter crop details in UI and get yield prediction instantly.  

## Model Performance  
The **Enhanced Random Forest Regressor** achieved:  
 **High R² Score** for better accuracy.  
 **Optimized ETL pipeline** for data preprocessing.  
 **Improved feature engineering** to enhance model predictions.  

## Acknowledgment  
- **Data Preprocessing & Cleaning**: Done in [ML_DataPreProcessing Repository](https://github.com/Anudeep007-hub/ML_DataPreProcessing.git).  
- **Machine Learning Models & Prediction**: Implemented in this repository.  

---

🔥 **This project provides an end-to-end ML pipeline for accurate crop yield forecasting!**  
🚀 Feel free to contribute & improve!  
```

### Improvements:  
 **Concise & Professional** - Covers everything clearly.  
 **Includes Your Second Repo Link** (for data preprocessing).  
 **Clean Structure** - Easy for others to understand & contribute.  
