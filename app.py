import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS with Font Awesome icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .icon-header {
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    .fa-icon {
        margin-right: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with icon
st.markdown('''
    <div class="main-header">
        <i class="fas fa-robot fa-icon"></i>
        ML Classification Dashboard
    </div>
''', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown('<h2><i class="fas fa-cog fa-icon"></i>Configuration</h2>', unsafe_allow_html=True)
st.sidebar.markdown("### Upload your test dataset and select a model")

# Model mapping
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest (Ensemble)': 'random_forest.pkl',
    'XGBoost (Ensemble)': 'xgboost.pkl'
}

@st.cache_resource
def load_model(model_path):
    """Load trained model from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load scaler from pickle file"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.warning("Scaler not found. Using default StandardScaler.")
        return StandardScaler()

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba,
                                              multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig

def plot_metrics_bar(metrics):
    """Plot metrics as bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# Main Application
def main():
    # Model selection dropdown with icon
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p><i class="fas fa-brain fa-icon"></i><b>Select Model</b></p>', unsafe_allow_html=True)
    selected_model_name = st.sidebar.selectbox(
        "Choose a classification model",
        list(MODEL_FILES.keys()),
        help="Choose a classification model",
        label_visibility="collapsed"
    )
    
    # File upload with icon
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p><i class="fas fa-file-upload fa-icon"></i><b>Upload Test Dataset (CSV)</b></p>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Upload your test data in CSV format",
        type=['csv'],
        help="Upload your test data in CSV format",
        label_visibility="collapsed"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header"><i class="fas fa-chart-bar fa-icon"></i>Dataset Information</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="sub-header"><i class="fas fa-check-circle fa-icon"></i>Selected: {selected_model_name}</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f" Dataset loaded successfully! Shape: {df.shape}")
            
            # Display dataset preview
            with st.expander(" View Dataset Preview"):
                st.dataframe(df.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
            
            # Target column selection with icon
            st.markdown("---")
            st.markdown('<p><i class="fas fa-bullseye fa-icon"></i><b>Select Target Column</b></p>', unsafe_allow_html=True)
            target_column = st.selectbox(
                "Choose the column containing true labels",
                df.columns.tolist(),
                help="Choose the column containing true labels",
                label_visibility="collapsed"
            )
            
            if target_column:
                # Prepare data
                X_test = df.drop(target_column, axis=1)
                y_test = df[target_column]
                
                # Load scaler and model
                scaler = load_scaler()
                model_path = f"model/{MODEL_FILES[selected_model_name]}"
                
                if os.path.exists(model_path):
                    model = load_model(model_path)
                    
                    if model is not None:
                        # Predict button with icon
                        if st.button("Run Prediction & Evaluation", type="primary", use_container_width=True):
                            with st.spinner("Running predictions..."):
                                try:
                                    # Scale features
                                    X_test_scaled = scaler.transform(X_test)
                                    
                                    # Make predictions
                                    y_pred = model.predict(X_test_scaled)
                                    y_pred_proba = model.predict_proba(X_test_scaled)
                                    
                                    # Calculate metrics
                                    metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
                                    
                                    st.success("Predictions completed successfully!")
                                    
                                    # Display metrics with icons
                                    st.markdown("---")
                                    st.markdown('<div class="sub-header"><i class="fas fa-chart-line fa-icon"></i>Evaluation Metrics</div>', unsafe_allow_html=True)
                                    
                                    # Metrics in columns
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                                    
                                    with col2:
                                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                                    
                                    with col3:
                                        st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                                        st.metric("MCC Score", f"{metrics['MCC']:.4f}")
                                    
                                    # Metrics bar chart
                                    st.markdown("---")
                                    fig_metrics = plot_metrics_bar(metrics)
                                    st.pyplot(fig_metrics)
                                    
                                    # Confusion Matrix with icon
                                    st.markdown("---")
                                    st.markdown('<div class="sub-header"><i class="fas fa-table fa-icon"></i>Confusion Matrix</div>', unsafe_allow_html=True)
                                    
                                    cm = confusion_matrix(y_test, y_pred)
                                    fig_cm = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model_name}")
                                    st.pyplot(fig_cm)
                                    
                                    # Classification Report with icon
                                    st.markdown("---")
                                    st.markdown('<div class="sub-header"><i class="fas fa-clipboard-list fa-icon"></i>Classification Report</div>', unsafe_allow_html=True)
                                    
                                    report = classification_report(y_test, y_pred, output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', axis=1))
                                    
                                    # Download predictions with icon
                                    st.markdown("---")
                                    predictions_df = pd.DataFrame({
                                        'True Label': y_test.values,
                                        'Predicted Label': y_pred
                                    })
                                    
                                    csv = predictions_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Predictions",
                                        data=csv,
                                        file_name=f'predictions_{selected_model_name.replace(" ", "_")}.csv',
                                        mime='text/csv'
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error during prediction: {e}")
                                    st.info("Please ensure your dataset has the same features as the training data.")
                else:
                    st.error(f"Model file not found: {model_path}")
                    st.info("Please ensure the model files are in the 'model/' directory.")
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.info("Please upload a valid CSV file.")
    
    else:
        # Instructions when no file is uploaded
        st.info("Please upload a CSV file to get started")
        
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-book fa-icon"></i>Instructions</h3>', unsafe_allow_html=True)
        st.markdown("""
        1. **Upload Dataset**: Click on 'Browse files' in the sidebar to upload your test dataset (CSV format)
        2. **Select Target Column**: Choose the column that contains the true labels
        3. **Select Model**: Choose one of the 6 classification models from the dropdown
        4. **Run Prediction**: Click the 'Run Prediction & Evaluation' button to see results
        5. **View Results**: Explore evaluation metrics, confusion matrix, and classification report
        """)
        
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-robot fa-icon"></i>Available Models</h3>', unsafe_allow_html=True)
        st.markdown("""
        - **Logistic Regression**: Linear classifier for binary/multiclass problems
        - **Decision Tree**: Tree-based non-parametric model
        - **K-Nearest Neighbor**: Instance-based learning algorithm
        - **Naive Bayes**: Probabilistic classifier based on Bayes theorem
        - **Random Forest**: Ensemble of decision trees (bagging)
        - **XGBoost**: Gradient boosting ensemble method
        """)

# Footer with icons
st.sidebar.markdown("---")
st.sidebar.markdown('<h3><i class="fas fa-user fa-icon"></i>About</h3>', unsafe_allow_html=True)
st.sidebar.info("""
**ML Assignment 2**  
BITS ID - 2025AB05025  
NAME - Mahender Singh
Machine Learning Classification  
""")

if __name__ == "__main__":
    main()
