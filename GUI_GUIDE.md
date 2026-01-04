# GUI User Guide

## AI Assistant Usage Behavior Analysis - Graphical Interface

### Getting Started

1. **Launch the Application**
   ```bash
   python gui.py
   ```

2. **Load Dataset**
   - Go to File > Load Dataset (or press Ctrl+O)
   - Select the CSV file containing your AI assistant usage data
   - The dataset path will be displayed in the status bar

3. **Run Analysis**
   - Go to Analysis > Run Analysis (or press Ctrl+R)
   - The analysis will run in a background thread
   - Progress updates will appear in the main window
   - Wait for the completion message

### Viewing Results

Once the analysis is complete, you can access the following features:

#### View Results Window (Ctrl+V)
Opens a tabbed interface with three sections:

1. **Statistics Tab**
   - Dataset shape and dimensions
   - Descriptive statistics for numerical columns
   - Frequency counts for categorical variables
   - All information is displayed in a scrollable text area

2. **Model Performance Tab**
   - Satisfaction Model metrics (accuracy)
   - Tokens Model metrics (MAE, RMSE, R2)
   - Session Length Model metrics (MAE, RMSE, R2)
   - Top feature importance for each model

3. **Insights Tab**
   - Device usage patterns
   - Model performance comparisons
   - Usage category analysis
   - Temporal patterns (hourly activity)

#### Make Prediction Window (Ctrl+P)
Interactive form for making predictions:

**Input Fields:**
- Device: Choose from Desktop, Mobile, Tablet, Smart Speaker
- Usage Category: Select from available categories in dataset
- Assistant Model: Choose from GPT-4o, GPT-5, GPT-5.1, o1, Mini
- Prompt Length: Enter value between 1-500
- Hour: Select hour of day (0-23)
- Day of Week: Choose from Monday-Sunday
- Month: Select month (1-12)

**Making a Prediction:**
1. Fill in all input fields
2. Click the "Predict" button
3. View results in the text area below:
   - Predicted satisfaction rating (1-5)
   - Estimated tokens to be used
   - Expected session length in minutes
   - Interpretation of results

### Menu Options

**File Menu**
- Load Dataset (Ctrl+O): Load a CSV dataset
- Exit (Ctrl+Q): Close the application

**Analysis Menu**
- Run Analysis (Ctrl+R): Execute full analysis pipeline
- View Results (Ctrl+V): Open results window
- Make Prediction (Ctrl+P): Open prediction window

**Help Menu**
- About: Display application information

### Generated Visualizations

The analysis generates PNG files in the project directory:
- numerical_distributions.png - Histograms of numerical features
- categorical_distributions.png - Bar charts of categorical variables
- correlation_matrix.png - Feature correlation heatmap
- satisfaction_by_device.png - Average satisfaction per device
- hourly_usage.png - Usage patterns by hour
- tokens_by_category.png - Token consumption by category
- session_by_model.png - Session lengths by AI model
- confusion_matrix_satisfaction.png - Satisfaction model performance
- feature_importance_satisfaction.png - Feature importance rankings
- tokens_prediction.png - Actual vs predicted tokens
- session_length_prediction.png - Actual vs predicted session length

### Tips

1. **Performance**: Analysis may take 30-60 seconds depending on dataset size
2. **Multiple Predictions**: You can make multiple predictions without re-running analysis
3. **Visualization**: Check the PNG files for detailed visual insights
4. **Errors**: Any errors during analysis will be displayed in message boxes

### Troubleshooting

**"Please load a dataset first" error**
- You need to load a CSV file before running analysis

**"Please run analysis first" error**
- Complete the analysis before viewing results or making predictions

**Application freezes during analysis**
- This is normal - the analysis runs in a background thread
- Wait for the completion message

**wxPython import error**
- Install wxPython: `pip install wxPython`

### System Requirements

- Python 3.8 or higher
- wxPython 4.2.0 or higher
- All dependencies from requirements.txt
- Recommended: 4GB RAM minimum for large datasets

### Keyboard Shortcuts

- Ctrl+O: Load Dataset
- Ctrl+R: Run Analysis
- Ctrl+V: View Results
- Ctrl+P: Make Prediction
- Ctrl+Q: Exit Application
