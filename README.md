# AI Assistant Usage Behavior Analysis

Comprehensive analysis and prediction tool for AI assistant usage patterns.

## Description

This project provides both command-line and GUI interfaces for analyzing AI assistant usage behavior. It uses Machine Learning to predict:
- User satisfaction ratings (1-5)
- Tokens consumed in a session
- Session duration in minutes

The analysis is based on the structure from the Kaggle notebook "Daily AI Assistant Usage Behavior" and provides extensive exploratory data analysis with visualizations.

## Features

### Analysis Components
1. **Satisfaction Predictor**: RandomForestClassifier for classifying satisfaction levels
2. **Tokens Predictor**: GradientBoostingRegressor for estimating token consumption
3. **Session Length Predictor**: RandomForestRegressor for estimating session duration

### Exploratory Data Analysis
- Distribution analysis for all numerical and categorical features
- Correlation matrix with heatmap visualization
- Device, model, and category performance comparisons
- Temporal pattern identification (hourly, daily, monthly)
- Comprehensive statistical summaries

### Generated Visualizations
- numerical_distributions.png - Histograms of all numerical features
- categorical_distributions.png - Bar charts of categorical variables
- correlation_matrix.png - Heatmap showing feature correlations
- satisfaction_by_device.png - Average satisfaction per device type
- hourly_usage.png - Usage patterns throughout the day
- tokens_by_category.png - Token consumption by usage category
- session_by_model.png - Session lengths by AI model
- confusion_matrix_satisfaction.png - Model performance matrix
- feature_importance_satisfaction.png - Most important features
- tokens_prediction.png - Actual vs predicted scatter plot
- session_length_prediction.png - Actual vs predicted scatter plot

## Usage

### Command-Line Interface

Run the complete analysis:

```bash
python main.py
```

This will:
1. Load and analyze the dataset
2. Train all three prediction models
3. Generate all visualizations (saved as PNG files)
4. Display key insights
5. Show example predictions

### Graphical User Interface

**Option 1: Tkinter GUI (Recommended for Linux)**

Launch the Tkinter-based GUI (more stable on Linux):

```bash
python gui_tk.py
```

**Features:**
- Auto-loads dataset from current directory
- Big START ANALYSIS button in the center
- Progress tracking with status updates
- Automatically saves 11 PNG visualizations to project folder
- View detailed results in organized tabs
- Make predictions with interactive form
- Native look and feel

**Option 2: wxPython GUI**

Launch the wxPython GUI (may have issues on some Linux distributions):

```bash
python gui.py
```

If you encounter GTK errors with wxPython, use the Tkinter version instead.

### Programmatic Usage

```python
from analysis import AIUsageBehaviorAnalysis

# Initialize
analysis = AIUsageBehaviorAnalysis('Daily_AI_Assistant_Usage_Behavior_Dataset.csv')

# Run complete analysis
analysis.run_full_analysis()

# Make custom predictions
prediction = analysis.predict(
    device='Desktop',
    usage_category='Coding',
    assistant_model='GPT-5',
    prompt_length=150,
    hour=14,
    day_of_week=2,
    month=1
)

print(f"Satisfaction: {prediction['satisfaction']}")
print(f"Tokens: {prediction['tokens']}")
print(f"Duration: {prediction['session_length']} minutes")
```

## Dataset Structure

The dataset includes the following features:
- **timestamp**: Date and time of the session
- **device**: Device type (Desktop, Mobile, Tablet, Smart Speaker)
- **usage_category**: Usage category (Coding, Research, Writing, Education, etc.)
- **prompt_length**: Length of the prompt
- **session_length_minutes**: Session duration
- **satisfaction_rating**: Satisfaction rating (1-5)
- **assistant_model**: AI model (GPT-4o, GPT-5, GPT-5.1, o1, Mini)
- **tokens_used**: Number of tokens consumed

## Dependencies

**Required (always install):**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

**Optional (for wxPython GUI):**
```
wxPython
```

Note: Tkinter comes pre-installed with Python, no installation needed.

Install required dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Install wxPython (optional, only if you want to use gui.py):
```bash
pip install wxPython
```

## Project Structure

```
proyecto-final/
├── main.py                                    # Command-line interface
├── analysis.py                                # Core analysis class
├── gui_tk.py                                  # Tkinter GUI (recommended for Linux)
├── gui.py                                     # wxPython GUI (alternative)
├── Daily_AI_Assistant_Usage_Behavior_Dataset.csv  # Dataset
├── README.md                                  # Documentation
└── *.png                                      # Generated visualizations
```

## Model Performance

### Satisfaction Prediction
- Metric: Accuracy
- Algorithm: Random Forest Classifier
- Features: Device, category, model, prompt length, temporal data

### Token Prediction
- Metrics: MAE, RMSE, R-squared
- Algorithm: Gradient Boosting Regressor
- Features: Same as satisfaction + satisfaction rating

### Session Length Prediction
- Metrics: MAE, RMSE, R-squared
- Algorithm: Random Forest Regressor
- Features: Same as token prediction

## Use Cases

1. **Experience Optimization**: Predict satisfaction before interaction
2. **Resource Management**: Estimate token consumption for planning
3. **Pattern Analysis**: Identify usage trends and patterns
4. **Recommendations**: Suggest optimal configurations based on predictions
5. **Performance Monitoring**: Track model and device performance

## Reference

Based on the Kaggle notebook structure:
"Daily AI Assistant Usage Behavior" by Ayse irem Colak

## Author

Proyecto Final - Inteligencia Artificial
ESCOM, 9 Semestre

