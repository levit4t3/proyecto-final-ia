# Project Summary

## AI Assistant Usage Behavior Analysis

### Overview
This project provides a comprehensive analysis and prediction system for AI assistant usage patterns. It includes both command-line and graphical user interfaces.

### Key Modifications Made

1. **Removed All Emojis**
   - All source code files (analysis.py, main.py, gui.py) are emoji-free
   - Professional, clean code suitable for production environments

2. **Kaggle Notebook Structure**
   - Based on "Daily AI Assistant Usage Behavior" notebook
   - Comprehensive exploratory data analysis
   - Multiple visualization outputs
   - Statistical summaries and insights
   - Three machine learning models for predictions

3. **Native GUI with wxPython**
   - Cross-platform native look and feel
   - Menu-driven interface
   - Progress tracking with background threads
   - Multiple windows for different functions
   - Keyboard shortcuts for efficiency

### Project Structure

```
proyecto-final/
├── analysis.py                    # Core analysis class with ML models
├── main.py                        # Command-line interface
├── gui.py                         # wxPython GUI application
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
├── GUI_GUIDE.md                   # GUI user guide
├── Daily_AI_Assistant_Usage_Behavior_Dataset.csv  # Dataset
└── *.png                          # Generated visualizations (11 files)
```

### Core Features

#### Analysis Module (analysis.py)
- AIUsageBehaviorAnalysis class
- Data loading and preprocessing
- Exploratory data analysis with 7+ visualizations
- Three ML models:
  - Random Forest Classifier (satisfaction)
  - Gradient Boosting Regressor (tokens)
  - Random Forest Regressor (session length)
- Feature importance analysis
- Prediction capabilities

#### Command-Line Interface (main.py)
- Simple execution: `python main.py`
- Runs complete analysis pipeline
- Generates all visualizations
- Provides example predictions
- Outputs results to console and PNG files

#### GUI Application (gui.py)
- Native wxPython interface
- Main window with status tracking
- Results viewer with tabbed interface
- Prediction form with interactive inputs
- Menu system with keyboard shortcuts
- Threading for non-blocking analysis
- Error handling and user feedback

### Generated Outputs

**Visualizations (PNG files):**
1. numerical_distributions.png - Distribution histograms
2. categorical_distributions.png - Category bar charts
3. correlation_matrix.png - Feature correlation heatmap
4. satisfaction_by_device.png - Device performance
5. hourly_usage.png - Temporal patterns
6. tokens_by_category.png - Token consumption
7. session_by_model.png - Model comparison
8. confusion_matrix_satisfaction.png - Model accuracy
9. feature_importance_satisfaction.png - Important features
10. tokens_prediction.png - Regression performance
11. session_length_prediction.png - Duration prediction

**Console Output:**
- Dataset statistics
- Model performance metrics
- Feature importance rankings
- Key insights and patterns
- Example predictions

### Usage Instructions

**Command-Line:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py
```

**GUI Application:**
```bash
# Launch GUI
python gui.py

# Then use menu:
# 1. File > Load Dataset
# 2. Analysis > Run Analysis
# 3. Analysis > View Results
# 4. Analysis > Make Prediction
```

### Technical Specifications

**Machine Learning:**
- Scikit-learn models with optimized hyperparameters
- 75-25 train-test split
- Classification and regression tasks
- Feature encoding for categorical variables
- Temporal feature extraction

**Data Processing:**
- Pandas for data manipulation
- NumPy for numerical operations
- Label encoding for categories
- Datetime parsing and feature extraction

**Visualization:**
- Matplotlib for plotting
- Seaborn for enhanced styling
- 300 DPI output for publication quality
- Multiple chart types (histogram, bar, scatter, heatmap)

**GUI Framework:**
- wxPython for native widgets
- Multi-threading for responsiveness
- Scrolled panels for large content
- Dialog boxes for file selection
- Status bars and progress indicators

### Dependencies

All packages with version requirements:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- wxPython >= 4.2.0

### Performance Characteristics

- Dataset: 300 records (expandable)
- Analysis time: 30-60 seconds
- Memory usage: ~100MB
- Output size: ~2MB (visualizations)
- Model training: 3 models in parallel

### Quality Assurance

- No emojis in source code (verified)
- Professional code formatting
- Comprehensive error handling
- User-friendly feedback
- Documentation included
- Example predictions provided

### Future Enhancements

Possible improvements:
1. Real-time prediction API
2. Database integration
3. Web-based dashboard
4. Model persistence (save/load)
5. Batch prediction mode
6. Additional visualization types
7. Export results to PDF/Excel

### Credits

- Dataset structure: Kaggle notebook by Ayse irem Colak
- Implementation: ESCOM 9th Semester Project
- Framework: Python, scikit-learn, wxPython
- Analysis methodology: Standard ML pipeline

### License

Educational project for academic purposes.
