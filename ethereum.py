import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load Dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    data = pd.read_csv("transaction_dataset.csv")  
    return data

# Function to train models and return predictions
def train_models(X_train, y_train, model_choice):
    if model_choice == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_choice == 'Random Forest':
        model = RandomForestClassifier()
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_choice == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    model.fit(X_train, y_train)
    return model

# Streamlit Application
st.title("Ethereum Fraud Detection")

# Load data
data = load_data()

# Show dataset overview
if st.checkbox('Show dataset'):
    st.write(data.head())

# Select features for training
st.sidebar.header("Feature Selection")
features = st.sidebar.multiselect("Select Features", data.columns[:-1])  # Exclude 'FLAG' (fraud label)

# Select model
model_choice = st.sidebar.selectbox("Select Model", ('Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost'))

# Split data into train and test
if features:
    X = data[features]
    y = data['FLAG']  # Assuming 'FLAG' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = train_models(X_train, y_train, model_choice)

    # Make predictions
    predictions = model.predict(X_test)

    # Show classification report
    st.subheader(f"{model_choice} Classification Report")
    report = classification_report(y_test, predictions, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    auc_score = auc(recall, precision)
    st.write(f"Area Under Precision-Recall Curve: {auc_score:.2f}")
    
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_choice} Precision-Recall Curve')
    st.pyplot(plt)

# User Input for Fraud Prediction
st.sidebar.header("User Input for Fraud Detection")
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(f"Input {feature}", value=0.0)

if st.sidebar.button("Predict Fraud"):
    user_input_df = pd.DataFrame([input_data])
    fraud_prediction = model.predict(user_input_df)[0]
    
    if fraud_prediction == 1:
        st.write("The transaction is potentially fraudulent!")
    else:
        st.write("The transaction is not fraudulent.")
