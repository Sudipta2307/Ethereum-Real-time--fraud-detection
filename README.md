# Ethereum Fraud Detection  

## Introduction  
Ethereum Fraud Detection is a machine learning-based project aimed at identifying fraudulent transactions on the Ethereum blockchain. This project leverages advanced data analytics and visualization to detect anomalies and classify fraudulent activities. It also includes a user-friendly **Streamlit application** for real-time interaction with the dataset and model predictions.  

## Features  
- **Fraud Detection Models**: Implements and compares multiple machine learning models for identifying fraud.  
- **Streamlit Application**: Provides an interactive dashboard for users to explore the dataset and analyze fraud detection results.  
- **Exploratory Data Analysis (EDA)**: Visualizes patterns and anomalies in Ethereum transaction data.  
- **Custom Features**: Extracts unique metrics from blockchain data, such as Ether and ERC20 token usage patterns.  

## Dataset  
The dataset consists of:  
- **Features**: Transaction indices, addresses, Ether metrics, ERC20 token data, transaction times, and unique identifiers.  
- **Source**: Publicly available Ethereum transaction datasets.  
- **Target Variable**: Label indicating whether a transaction is fraudulent (1) or legitimate (0).  

## Technologies Used  
- **Python**: Primary programming language for data processing and model development.  
- **Streamlit**: For building the interactive user interface.  
- **Libraries/Frameworks**:  
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`, `plotly`  
  - Machine Learning: `scikit-learn`, `XGBoost`  
  - Deployment: `Streamlit`  

## Project Workflow  
1. **Data Understanding and Preprocessing**:  
   - Data cleaning and transformation.  
   - Feature engineering to create meaningful metrics.  

2. **Exploratory Data Analysis (EDA)**:  
   - Analyze transaction patterns.  
   - Identify features influencing fraud detection.  

3. **Model Development**:  
   - Train and evaluate machine learning models (e.g., Logistic Regression, XGBoost, Decision Trees).  
   - Optimize models for imbalanced data using techniques like SMOTE and custom sampling methods.  

4. **Application Development**:  
   - Build a Streamlit app to allow users to upload data and receive fraud detection insights.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/ethereum-fraud-detection.git  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the Streamlit app:  
   ```bash  
   streamlit run ethereum.py  
   ```  

## Usage  
1. Open the Streamlit app in your browser.  
2. Upload a dataset or use the provided sample dataset.  
3. View fraud detection results and explore EDA insights.  

## Results  
- The best-performing model achieved an accuracy of **X%** and an AUPRC of **Y%**.  
- Fraudulent transactions were identified with high precision and recall.  

## Future Enhancements  
- Integrate real-time data ingestion from blockchain APIs.  
- Implement deep learning models for improved accuracy.  
- Add a feature to predict fraudulent transactions before they are confirmed.  

## Contributing  
Contributions are welcome! Please fork this repository and submit a pull request with your proposed changes.  
