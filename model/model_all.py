# ML Assignment 2 - Model Training
# Author: Mahender Singh
# BITS ID: 2025AB05025
# Dataset: Credit Card Fraud Detection https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                              recall_score, f1_score, matthews_corrcoef,
                              confusion_matrix, classification_report)
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

class MLClassificationPipeline:
    """
    Assignment 2
    Implements 6 classification models with comprehensive evaluation
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self, target_column, test_size=0.2, random_state=42):
        """
        Load dataset and perform train-test split
        """
        print("Loading dataset...")
        df = pd.read_csv(self.dataset_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Instances: {df.shape[0]}")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Save test data to test.csv for Streamlit app
        test_df = self.X_test.copy()
        test_df[target_column] = self.y_test.values
        test_path = "test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"Saved test data to: {test_path}")
        
        # Feature scaling
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """
        Initializing all 6 classification models
        """
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
        }
        print("Models initialized successfully!")
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculating all required evaluation metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # AUC Score (requires probability predictions)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Multi-class
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
            except Exception as e:
                metrics['AUC'] = 0.0
        else:
            metrics['AUC'] = 0.0
            
        return metrics
    
    def train_and_evaluate_all(self):
        """
        Training all models and calculating evaluation metrics
        """
        print("\n" + "="*70)
        print("TRAINING AND EVALUATION")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Training: {model_name}")
            print(f"{'='*70}")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred)
            }
            
            # Print metrics
            print(f"\nEvaluation Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
                
        print("\n" + "="*70)
        print("All models trained successfully!")
        print("="*70)
        
    def display_results_table(self):
        """
        Displaying comparison table of all models
        """
        print("\n" + "="*100)
        print("MODEL COMPARISON TABLE")
        print("="*100)
        
        # Create DataFrame for results
        results_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            results_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']:.4f}",
                'AUC': f"{metrics['AUC']:.4f}",
                'Precision': f"{metrics['Precision']:.4f}",
                'Recall': f"{metrics['Recall']:.4f}",
                'F1 Score': f"{metrics['F1 Score']:.4f}",
                'MCC': f"{metrics['MCC']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        print("="*100)
        
        return results_df
    
    def save_models(self, output_dir='model'):
        """
        Saving trained models and scaler
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, result in self.results.items():
            model_filename = f"{output_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(result['model'], f)
            print(f"Saved: {model_filename}")
        
        # Save scaler
        scaler_filename = f"{output_dir}/scaler.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {scaler_filename}")
        
        # Save results
        results_filename = f"{output_dir}/results.pkl"
        with open(results_filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Saved: {results_filename}")
        
        print("\nAll models saved successfully!")
        
    


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("ML ASSIGNMENT 2 - CLASSIFICATION MODELS")
    print("="*70)
    
    # Initialize pipeline
    
    pipeline = MLClassificationPipeline('creditcard.csv')
    
    # Load and preprocess data
    pipeline.load_and_preprocess_data(target_column='Class')
    
    # Initialize models
    pipeline.initialize_models()
    
    # Train and evaluate
    pipeline.train_and_evaluate_all()
    
    # Display results
    results_df = pipeline.display_results_table()
    
    
    
    
    # Save models
    pipeline.save_models()
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
