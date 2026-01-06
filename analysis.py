import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                            mean_absolute_error, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


class AIUsageBehaviorAnalysis:
    """
    Complete analysis class for AI Assistant Usage Behavior Dataset
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset"""
        print("="*70)
        print("LOADING DATASET")
        print("="*70)
        self.df = pd.read_csv(self.dataset_path)
        
        # Parse timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Extract time features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['day'] = self.df['timestamp'].dt.day
        
        print(f"Dataset loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        print(f"\nDataset Info:")
        print(self.df.info())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics"""
        print("\n" + "="*70)
        print("BASIC STATISTICS")
        print("="*70)
        print("\nDescriptive Statistics:")
        print(self.df.describe())
        
        print("\n\nCategorical Variables Distribution:")
        categorical_cols = ['device', 'usage_category', 'assistant_model']
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
            
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA with visualizations"""
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        # 1. Distribution of numerical features
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
        
        self.df['prompt_length'].hist(bins=30, ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title('Prompt Length Distribution')
        axes[0, 0].set_xlabel('Prompt Length')
        axes[0, 0].set_ylabel('Frequency')
        
        self.df['session_length_minutes'].hist(bins=30, ax=axes[0, 1], edgecolor='black')
        axes[0, 1].set_title('Session Length Distribution')
        axes[0, 1].set_xlabel('Session Length (minutes)')
        axes[0, 1].set_ylabel('Frequency')
        
        self.df['tokens_used'].hist(bins=30, ax=axes[1, 0], edgecolor='black')
        axes[1, 0].set_title('Tokens Used Distribution')
        axes[1, 0].set_xlabel('Tokens Used')
        axes[1, 0].set_ylabel('Frequency')
        
        self.df['satisfaction_rating'].hist(bins=5, ax=axes[1, 1], edgecolor='black')
        axes[1, 1].set_title('Satisfaction Rating Distribution')
        axes[1, 1].set_xlabel('Satisfaction Rating')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')
        print("\nSaved: numerical_distributions.png")
        
        # 2. Categorical distributions
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Categorical Variables Distribution', fontsize=16, fontweight='bold')
        
        self.df['device'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title('Device Distribution')
        axes[0].set_xlabel('Device')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        self.df['usage_category'].value_counts().plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black')
        axes[1].set_title('Usage Category Distribution')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
        
        self.df['assistant_model'].value_counts().plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='black')
        axes[2].set_title('Assistant Model Distribution')
        axes[2].set_xlabel('Model')
        axes[2].set_ylabel('Count')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
        print("Saved: categorical_distributions.png")
        
        # 3. Correlation matrix
        plt.figure(figsize=(10, 8))
        numerical_cols = ['prompt_length', 'session_length_minutes', 'satisfaction_rating', 
                         'tokens_used', 'hour', 'day_of_week', 'month']
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: correlation_matrix.png")
        
        # 4. Satisfaction by device
        plt.figure(figsize=(10, 6))
        satisfaction_by_device = self.df.groupby('device')['satisfaction_rating'].mean().sort_values(ascending=False)
        satisfaction_by_device.plot(kind='barh', color='steelblue', edgecolor='black')
        plt.title('Average Satisfaction Rating by Device', fontsize=14, fontweight='bold')
        plt.xlabel('Average Satisfaction Rating')
        plt.ylabel('Device')
        plt.tight_layout()
        plt.savefig('satisfaction_by_device.png', dpi=300, bbox_inches='tight')
        print("Saved: satisfaction_by_device.png")
        
        # 5. Usage patterns by hour
        plt.figure(figsize=(12, 6))
        hourly_usage = self.df.groupby('hour').size()
        hourly_usage.plot(kind='line', marker='o', color='darkgreen', linewidth=2)
        plt.title('Usage Patterns by Hour of Day', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Sessions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hourly_usage.png', dpi=300, bbox_inches='tight')
        print("Saved: hourly_usage.png")
        
        # 6. Tokens used by category
        plt.figure(figsize=(10, 6))
        tokens_by_category = self.df.groupby('usage_category')['tokens_used'].mean().sort_values(ascending=False)
        tokens_by_category.plot(kind='barh', color='orange', edgecolor='black')
        plt.title('Average Tokens Used by Usage Category', fontsize=14, fontweight='bold')
        plt.xlabel('Average Tokens Used')
        plt.ylabel('Usage Category')
        plt.tight_layout()
        plt.savefig('tokens_by_category.png', dpi=300, bbox_inches='tight')
        print("Saved: tokens_by_category.png")
        
        # 7. Session length by model
        plt.figure(figsize=(10, 6))
        session_by_model = self.df.groupby('assistant_model')['session_length_minutes'].mean().sort_values(ascending=False)
        session_by_model.plot(kind='bar', color='purple', edgecolor='black')
        plt.title('Average Session Length by Assistant Model', fontsize=14, fontweight='bold')
        plt.xlabel('Assistant Model')
        plt.ylabel('Average Session Length (minutes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('session_by_model.png', dpi=300, bbox_inches='tight')
        print("Saved: session_by_model.png")
        
    def prepare_features(self, target_column, exclude_columns=None):
        """Prepare features for modeling"""
        df_model = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['device', 'usage_category', 'assistant_model']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_model[col + '_encoded'] = self.label_encoders[col].fit_transform(df_model[col])
            else:
                df_model[col + '_encoded'] = self.label_encoders[col].transform(df_model[col])
        
        # Select features
        feature_cols = [col + '_encoded' for col in categorical_cols] + \
                      ['prompt_length', 'hour', 'day_of_week', 'month']
        
        # Exclude specified columns if any
        if exclude_columns:
            feature_cols = [col for col in feature_cols if col not in exclude_columns]
        
        X = df_model[feature_cols]
        y = df_model[target_column]
        
        return X, y, feature_cols
    
    def train_satisfaction_model(self):
        """Train model to predict satisfaction rating"""
        print("\n" + "="*70)
        print("TRAINING SATISFACTION PREDICTION MODEL")
        print("="*70)
        
        X, y, feature_cols = self.prepare_features('satisfaction_rating')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Random Forest Classifier
        print("\nTraining Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title('Confusion Matrix - Satisfaction Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_satisfaction.png', dpi=300, bbox_inches='tight')
        print("\nSaved: confusion_matrix_satisfaction.png")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        plt.figure(figsize=(10, 6))
        importance_df.plot(x='feature', y='importance', kind='barh', color='teal', edgecolor='black', legend=False)
        plt.title('Feature Importance - Satisfaction Model', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance_satisfaction.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance_satisfaction.png")
        
        self.models['satisfaction'] = rf_model
        self.results['satisfaction'] = {
            'accuracy': accuracy,
            'feature_importance': importance_df
        }
        
        return rf_model, accuracy
    
    def train_tokens_model(self):
        """Train model to predict tokens used"""
        print("\n" + "="*70)
        print("TRAINING TOKENS PREDICTION MODEL")
        print("="*70)
        
        X, y, feature_cols = self.prepare_features('tokens_used')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Gradient Boosting Regressor
        print("\nTraining Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        gb_model.fit(X_train, y_train)
        y_pred = gb_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nMean Absolute Error (MAE): {mae:.2f} tokens")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} tokens")
        print(f"R-squared (R2): {r2:.4f}")
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Tokens Used')
        plt.ylabel('Predicted Tokens Used')
        plt.title('Actual vs Predicted Tokens Used', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('tokens_prediction.png', dpi=300, bbox_inches='tight')
        print("\nSaved: tokens_prediction.png")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        self.models['tokens'] = gb_model
        self.results['tokens'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': importance_df
        }
        
        return gb_model, r2
    
    def train_session_length_model(self):
        """Train model to predict session length"""
        print("\n" + "="*70)
        print("TRAINING SESSION LENGTH PREDICTION MODEL")
        print("="*70)
        
        X, y, feature_cols = self.prepare_features('session_length_minutes')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nMean Absolute Error (MAE): {mae:.2f} minutes")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
        print(f"R-squared (R2): {r2:.4f}")
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Session Length (minutes)')
        plt.ylabel('Predicted Session Length (minutes)')
        plt.title('Actual vs Predicted Session Length', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('session_length_prediction.png', dpi=300, bbox_inches='tight')
        print("\nSaved: session_length_prediction.png")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(importance_df)
        
        self.models['session_length'] = rf_model
        self.results['session_length'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': importance_df
        }
        
        return rf_model, r2
    
    def generate_insights(self):
        """Generate insights from the analysis"""
        print("\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)
        
        print("\n1. Device Usage Patterns:")
        print(self.df.groupby('device').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2))
        
        print("\n2. Model Performance:")
        print(self.df.groupby('assistant_model').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2))
        
        print("\n3. Usage Category Analysis:")
        print(self.df.groupby('usage_category').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2))
        
        print("\n4. Temporal Patterns:")
        print("\nMost active hours:")
        print(self.df.groupby('hour').size().sort_values(ascending=False).head())
        
        print("\nMost active days of week:")
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.df.groupby('day_of_week').size()
        for idx, count in day_counts.items():
            print(f"{days[idx]}: {count}")
    
    def predict(self, device, usage_category, assistant_model, prompt_length, hour, day_of_week, month):
        """Make predictions for given inputs"""
        print("\n" + "="*70)
        print("MAKING PREDICTION")
        print("="*70)
        
        # Encode inputs
        device_enc = self.label_encoders['device'].transform([device])[0]
        usage_enc = self.label_encoders['usage_category'].transform([usage_category])[0]
        model_enc = self.label_encoders['assistant_model'].transform([assistant_model])[0]
        
        X_input = np.array([[device_enc, usage_enc, model_enc, prompt_length, hour, day_of_week, month]])
        
        # Predictions
        satisfaction_pred = self.models['satisfaction'].predict(X_input)[0]
        tokens_pred = self.models['tokens'].predict(X_input)[0]
        session_pred = self.models['session_length'].predict(X_input)[0]
        
        print(f"\nInput Parameters:")
        print(f"  Device: {device}")
        print(f"  Usage Category: {usage_category}")
        print(f"  Assistant Model: {assistant_model}")
        print(f"  Prompt Length: {prompt_length}")
        print(f"  Hour: {hour}")
        print(f"  Day of Week: {day_of_week}")
        print(f"  Month: {month}")
        
        print(f"\nPredictions:")
        print(f"  Satisfaction Rating: {satisfaction_pred}")
        print(f"  Tokens Used: {int(tokens_pred)}")
        print(f"  Session Length: {session_pred:.2f} minutes")
        
        return {
            'satisfaction': satisfaction_pred,
            'tokens': int(tokens_pred),
            'session_length': round(session_pred, 2)
        }
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.load_data()
        self.basic_statistics()
        self.exploratory_data_analysis()
        self.train_satisfaction_model()
        self.train_tokens_model()
        self.train_session_length_model()
        self.generate_insights()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("AI ASSISTANT USAGE BEHAVIOR ANALYSIS")
    print("="*70)
    
    # Initialize and run analysis
    analysis = AIUsageBehaviorAnalysis('Daily_AI_Assistant_Usage_Behavior_Dataset.csv')
    analysis.run_full_analysis()
    
    # Example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)
    
    # Example 1
    print("\nExample 1: Desktop Coding Session")
    analysis.predict(
        device='Desktop',
        usage_category='Coding',
        assistant_model='GPT-5',
        prompt_length=150,
        hour=14,
        day_of_week=2,
        month=1
    )
    
    # Example 2
    print("\nExample 2: Mobile Research Session")
    analysis.predict(
        device='Mobile',
        usage_category='Research',
        assistant_model='o1',
        prompt_length=80,
        hour=20,
        day_of_week=5,
        month=1
    )
    
    # Example 3
    print("\nExample 3: Tablet Writing Session")
    analysis.predict(
        device='Tablet',
        usage_category='Writing',
        assistant_model='GPT-4o',
        prompt_length=200,
        hour=10,
        day_of_week=1,
        month=1
    )


if __name__ == '__main__':
    main()
