import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data, scaler=None, training=False):
    if training:
        scaler = StandardScaler()
        data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])
    else:
        data[['Time', 'Amount']] = scaler.transform(data[['Time', 'Amount']])
    return data, scaler

def train_model(data):
    X = data.drop(columns=['Class'])
    y = data['Class']
    X, scaler = preprocess_data(X, training=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        use_label_encoder=True,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, scaler, accuracy

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    return plt

def predict_fraud(model, scaler, input_file):
    new_data = pd.read_csv(input_file)
    original_data = new_data.copy()
    
    if 'Class' in new_data.columns:
        y_true = new_data['Class']
    else:
        y_true = None
    
    new_data.drop(columns=['Class'], inplace=True, errors='ignore')
    new_data, _ = preprocess_data(new_data, scaler=scaler, training=False)
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]
    new_data['Prediction'] = predictions
    fraud_cases = new_data[new_data['Prediction'] == 1]
    fraud_cases_original = original_data.loc[fraud_cases.index]
    
    if y_true is not None:
        f1 = f1_score(y_true, predictions)
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        conf_matrix = confusion_matrix(y_true, predictions)
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        roc_auc = auc(fpr, tpr)
    else:
        f1 = accuracy = precision = recall = conf_matrix = fpr = tpr = roc_auc = None
    
    return fraud_cases_original, f1, accuracy, precision, recall, conf_matrix, fpr, tpr, roc_auc

def main():
    st.title("Fraud Detection with XGBoost")
    
    st.header("Step 1: Train the Model")
    uploaded_train_file = st.file_uploader("Upload Training CSV File", type=["csv"])
    if uploaded_train_file is not None:
        with st.spinner("Training the model..."):
            train_data = pd.read_csv(uploaded_train_file)
            model, scaler, accuracy = train_model(train_data)
            joblib.dump(model, 'xgboost_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")
    
    st.header("Step 2: Detect Fraud")
    uploaded_test_file = st.file_uploader("Upload Test CSV File", type=["csv"])
    if uploaded_test_file is not None:
        with st.spinner("Predicting fraud cases..."):
            model = joblib.load('xgboost_model.pkl') 
            scaler = joblib.load('scaler.pkl')
            
            results = predict_fraud(model, scaler, uploaded_test_file)
            fraudulent_samples, f1, acc, precision, recall, conf_matrix, fpr, tpr, roc_auc = results
            
            if not fraudulent_samples.empty:
                st.write("Fraudulent Samples Detected:")
                st.dataframe(fraudulent_samples)
            else:
                st.write("No fraudulent samples detected.")
            
            if all(metric is not None for metric in [f1, acc, precision, recall]):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{acc:.2f}")
                    st.metric("Precision", f"{precision:.2f}")
                with col2:
                    st.metric("Recall", f"{recall:.2f}")
                    st.metric("F1 Score", f"{f1:.2f}")
                
                st.subheader("Confusion Matrix")
                conf_fig = plot_confusion_matrix(conf_matrix)
                st.pyplot(conf_fig)
                plt.clf()
                
                st.subheader("ROC Curve")
                roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
                st.pyplot(roc_fig)
                plt.clf()
            else:
                st.warning("Metrics cannot be computed as 'Class' labels are missing in the test data.")

if __name__ == "__main__":
    main()