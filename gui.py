"""
GUI for AI Assistant Usage Behavior Analysis
Native look and feel using wxPython
"""

import wx
import wx.lib.scrolledpanel as scrolled
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
import threading
import sys
import os

# Import analysis module
from analysis import AIUsageBehaviorAnalysis


class AnalysisThread(threading.Thread):
    """Thread for running analysis in background"""
    def __init__(self, parent, dataset_path):
        threading.Thread.__init__(self)
        self.parent = parent
        self.dataset_path = dataset_path
        self.analysis = None
        
    def run(self):
        """Run analysis"""
        try:
            wx.CallAfter(self.parent.update_status, "Loading dataset...")
            self.analysis = AIUsageBehaviorAnalysis(self.dataset_path)
            self.analysis.load_data()
            
            wx.CallAfter(self.parent.update_status, "Running exploratory analysis...")
            self.analysis.basic_statistics()
            self.analysis.exploratory_data_analysis()
            
            wx.CallAfter(self.parent.update_status, "Training satisfaction model...")
            self.analysis.train_satisfaction_model()
            
            wx.CallAfter(self.parent.update_status, "Training tokens model...")
            self.analysis.train_tokens_model()
            
            wx.CallAfter(self.parent.update_status, "Training session length model...")
            self.analysis.train_session_length_model()
            
            wx.CallAfter(self.parent.update_status, "Generating insights...")
            self.analysis.generate_insights()
            
            wx.CallAfter(self.parent.on_analysis_complete, self.analysis)
        except Exception as e:
            wx.CallAfter(self.parent.on_analysis_error, str(e))


class ResultsFrame(wx.Frame):
    """Frame to display analysis results"""
    def __init__(self, parent, analysis):
        wx.Frame.__init__(self, parent, title="Analysis Results", size=(900, 700))
        self.analysis = analysis
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Notebook for tabs
        notebook = wx.Notebook(panel)
        
        # Statistics tab
        stats_panel = self.create_statistics_panel(notebook)
        notebook.AddPage(stats_panel, "Statistics")
        
        # Model Performance tab
        models_panel = self.create_models_panel(notebook)
        notebook.AddPage(models_panel, "Model Performance")
        
        # Insights tab
        insights_panel = self.create_insights_panel(notebook)
        notebook.AddPage(insights_panel, "Insights")
        
        main_sizer.Add(notebook, 1, wx.EXPAND | wx.ALL, 5)
        panel.SetSizer(main_sizer)
        
    def create_statistics_panel(self, parent):
        """Create statistics panel"""
        panel = scrolled.ScrolledPanel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Dataset info
        info_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, size=(-1, 200))
        info_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        info_str = f"Dataset Shape: {self.analysis.df.shape[0]} rows x {self.analysis.df.shape[1]} columns\n\n"
        info_str += "Descriptive Statistics:\n"
        info_str += str(self.analysis.df.describe()) + "\n\n"
        info_str += "Categorical Variables:\n"
        for col in ['device', 'usage_category', 'assistant_model']:
            info_str += f"\n{col}:\n{self.analysis.df[col].value_counts()}\n"
        
        info_text.SetValue(info_str)
        sizer.Add(wx.StaticText(panel, label="Dataset Statistics"), 0, wx.ALL, 5)
        sizer.Add(info_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        panel.SetupScrolling()
        return panel
    
    def create_models_panel(self, parent):
        """Create model performance panel"""
        panel = scrolled.ScrolledPanel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Satisfaction model
        sizer.Add(wx.StaticText(panel, label="Satisfaction Model (Random Forest Classifier)"), 
                 0, wx.ALL | wx.ALIGN_LEFT, 5)
        sat_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        sat_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sat_info = f"Accuracy: {self.analysis.results['satisfaction']['accuracy']:.4f}\n\n"
        sat_info += "Top Features:\n"
        sat_info += str(self.analysis.results['satisfaction']['feature_importance'].head())
        sat_text.SetValue(sat_info)
        sizer.Add(sat_text, 0, wx.EXPAND | wx.ALL, 5)
        
        # Tokens model
        sizer.Add(wx.StaticText(panel, label="Tokens Model (Gradient Boosting Regressor)"), 
                 0, wx.ALL | wx.ALIGN_LEFT, 5)
        tok_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        tok_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        tok_info = f"MAE: {self.analysis.results['tokens']['mae']:.2f}\n"
        tok_info += f"RMSE: {self.analysis.results['tokens']['rmse']:.2f}\n"
        tok_info += f"R2: {self.analysis.results['tokens']['r2']:.4f}\n\n"
        tok_info += "Top Features:\n"
        tok_info += str(self.analysis.results['tokens']['feature_importance'].head())
        tok_text.SetValue(tok_info)
        sizer.Add(tok_text, 0, wx.EXPAND | wx.ALL, 5)
        
        # Session length model
        sizer.Add(wx.StaticText(panel, label="Session Length Model (Random Forest Regressor)"), 
                 0, wx.ALL | wx.ALIGN_LEFT, 5)
        sess_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 100))
        sess_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        sess_info = f"MAE: {self.analysis.results['session_length']['mae']:.2f}\n"
        sess_info += f"RMSE: {self.analysis.results['session_length']['rmse']:.2f}\n"
        sess_info += f"R2: {self.analysis.results['session_length']['r2']:.4f}\n\n"
        sess_info += "Top Features:\n"
        sess_info += str(self.analysis.results['session_length']['feature_importance'].head())
        sess_text.SetValue(sess_info)
        sizer.Add(sess_text, 0, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        panel.SetupScrolling()
        return panel
    
    def create_insights_panel(self, parent):
        """Create insights panel"""
        panel = scrolled.ScrolledPanel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        insights_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2, size=(-1, 500))
        insights_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        insights_str = "KEY INSIGHTS\n" + "="*70 + "\n\n"
        
        insights_str += "1. Device Usage Patterns:\n"
        insights_str += str(self.analysis.df.groupby('device').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "2. Model Performance:\n"
        insights_str += str(self.analysis.df.groupby('assistant_model').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "3. Usage Category Analysis:\n"
        insights_str += str(self.analysis.df.groupby('usage_category').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "4. Temporal Patterns:\n"
        insights_str += "Most active hours:\n"
        insights_str += str(self.analysis.df.groupby('hour').size().sort_values(ascending=False).head()) + "\n"
        
        insights_text.SetValue(insights_str)
        sizer.Add(wx.StaticText(panel, label="Analysis Insights"), 0, wx.ALL, 5)
        sizer.Add(insights_text, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        panel.SetupScrolling()
        return panel


class PredictionFrame(wx.Frame):
    """Frame for making predictions"""
    def __init__(self, parent, analysis):
        wx.Frame.__init__(self, parent, title="Make Prediction", size=(600, 500))
        self.analysis = analysis
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Input form
        form_sizer = wx.FlexGridSizer(7, 2, 10, 10)
        form_sizer.AddGrowableCol(1)
        
        # Device
        form_sizer.Add(wx.StaticText(panel, label="Device:"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.device_choice = wx.Choice(panel, choices=['Desktop', 'Mobile', 'Tablet', 'Smart Speaker'])
        self.device_choice.SetSelection(0)
        form_sizer.Add(self.device_choice, 0, wx.EXPAND)
        
        # Usage Category
        form_sizer.Add(wx.StaticText(panel, label="Usage Category:"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        categories = list(self.analysis.df['usage_category'].unique())
        self.category_choice = wx.Choice(panel, choices=categories)
        self.category_choice.SetSelection(0)
        form_sizer.Add(self.category_choice, 0, wx.EXPAND)
        
        # Assistant Model
        form_sizer.Add(wx.StaticText(panel, label="Assistant Model:"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        models = list(self.analysis.df['assistant_model'].unique())
        self.model_choice = wx.Choice(panel, choices=models)
        self.model_choice.SetSelection(0)
        form_sizer.Add(self.model_choice, 0, wx.EXPAND)
        
        # Prompt Length
        form_sizer.Add(wx.StaticText(panel, label="Prompt Length:"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.prompt_spin = wx.SpinCtrl(panel, min=1, max=500, initial=100)
        form_sizer.Add(self.prompt_spin, 0, wx.EXPAND)
        
        # Hour
        form_sizer.Add(wx.StaticText(panel, label="Hour (0-23):"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.hour_spin = wx.SpinCtrl(panel, min=0, max=23, initial=12)
        form_sizer.Add(self.hour_spin, 0, wx.EXPAND)
        
        # Day of Week
        form_sizer.Add(wx.StaticText(panel, label="Day of Week:"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.day_choice = wx.Choice(panel, choices=days)
        self.day_choice.SetSelection(0)
        form_sizer.Add(self.day_choice, 0, wx.EXPAND)
        
        # Month
        form_sizer.Add(wx.StaticText(panel, label="Month (1-12):"), 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        self.month_spin = wx.SpinCtrl(panel, min=1, max=12, initial=1)
        form_sizer.Add(self.month_spin, 0, wx.EXPAND)
        
        main_sizer.Add(form_sizer, 0, wx.ALL | wx.EXPAND, 10)
        
        # Predict button
        predict_btn = wx.Button(panel, label="Predict")
        predict_btn.Bind(wx.EVT_BUTTON, self.on_predict)
        main_sizer.Add(predict_btn, 0, wx.ALL | wx.CENTER, 10)
        
        # Results
        main_sizer.Add(wx.StaticText(panel, label="Prediction Results:"), 0, wx.ALL, 10)
        self.results_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        self.results_text.SetFont(wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        main_sizer.Add(self.results_text, 1, wx.ALL | wx.EXPAND, 10)
        
        panel.SetSizer(main_sizer)
        
    def on_predict(self, event):
        """Handle predict button click"""
        try:
            device = self.device_choice.GetStringSelection()
            category = self.category_choice.GetStringSelection()
            model = self.model_choice.GetStringSelection()
            prompt_length = self.prompt_spin.GetValue()
            hour = self.hour_spin.GetValue()
            day_of_week = self.day_choice.GetSelection()
            month = self.month_spin.GetValue()
            
            result = self.analysis.predict(device, category, model, prompt_length, hour, day_of_week, month)
            
            result_str = "PREDICTION RESULTS\n" + "="*50 + "\n\n"
            result_str += f"Satisfaction Rating: {result['satisfaction']}/5\n"
            result_str += f"Tokens Used: {result['tokens']} tokens\n"
            result_str += f"Session Length: {result['session_length']} minutes\n\n"
            
            if result['satisfaction'] >= 4:
                result_str += "Interpretation: High satisfaction expected\n"
            elif result['satisfaction'] >= 3:
                result_str += "Interpretation: Neutral experience expected\n"
            else:
                result_str += "Interpretation: Low satisfaction - review parameters\n"
            
            self.results_text.SetValue(result_str)
            
        except Exception as e:
            wx.MessageBox(f"Error making prediction: {str(e)}", "Error", wx.ICON_ERROR)


class MainFrame(wx.Frame):
    """Main application frame"""
    def __init__(self):
        wx.Frame.__init__(self, None, title="AI Assistant Usage Behavior Analysis", size=(800, 600))
        self.analysis = None
        self.init_ui()
        self.Centre()
        
    def init_ui(self):
        """Initialize UI"""
        # Create menu bar
        menubar = wx.MenuBar()
        
        file_menu = wx.Menu()
        load_item = file_menu.Append(wx.ID_OPEN, "Load Dataset\tCtrl+O", "Load dataset")
        file_menu.AppendSeparator()
        exit_item = file_menu.Append(wx.ID_EXIT, "Exit\tCtrl+Q", "Exit application")
        menubar.Append(file_menu, "&File")
        
        analysis_menu = wx.Menu()
        self.run_item = analysis_menu.Append(wx.ID_ANY, "Run Analysis\tCtrl+R", "Run full analysis")
        self.run_item.Enable(False)
        self.results_item = analysis_menu.Append(wx.ID_ANY, "View Results\tCtrl+V", "View analysis results")
        self.results_item.Enable(False)
        self.predict_item = analysis_menu.Append(wx.ID_ANY, "Make Prediction\tCtrl+P", "Make prediction")
        self.predict_item.Enable(False)
        menubar.Append(analysis_menu, "&Analysis")
        
        help_menu = wx.Menu()
        about_item = help_menu.Append(wx.ID_ABOUT, "About", "About this application")
        menubar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menubar)
        
        # Bind menu events
        self.Bind(wx.EVT_MENU, self.on_load, load_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)
        self.Bind(wx.EVT_MENU, self.on_run_analysis, self.run_item)
        self.Bind(wx.EVT_MENU, self.on_view_results, self.results_item)
        self.Bind(wx.EVT_MENU, self.on_make_prediction, self.predict_item)
        self.Bind(wx.EVT_MENU, self.on_about, about_item)
        
        # Create panel
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="AI Assistant Usage Behavior Analysis")
        title_font = wx.Font(16, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.CENTER, 20)
        
        # Instructions
        instructions = wx.StaticText(panel, label=
            "Welcome to the AI Assistant Usage Behavior Analysis Tool\n\n"
            "Instructions:\n"
            "1. Load your dataset using File > Load Dataset\n"
            "2. Run the analysis using Analysis > Run Analysis\n"
            "3. View results and make predictions\n\n"
            "Features:\n"
            "- Comprehensive exploratory data analysis\n"
            "- Machine learning model training\n"
            "- Satisfaction, tokens, and session length prediction\n"
            "- Visual insights and reports"
        )
        main_sizer.Add(instructions, 0, wx.ALL | wx.CENTER, 20)
        
        # Status text
        self.status_text = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 200))
        self.status_text.SetFont(wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        main_sizer.Add(self.status_text, 1, wx.ALL | wx.EXPAND, 20)
        
        # Progress bar
        self.progress = wx.Gauge(panel, range=100)
        main_sizer.Add(self.progress, 0, wx.ALL | wx.EXPAND, 20)
        
        panel.SetSizer(main_sizer)
        
        # Status bar
        self.CreateStatusBar()
        self.SetStatusText("Ready")
        
    def on_load(self, event):
        """Handle load dataset"""
        wildcard = "CSV files (*.csv)|*.csv"
        dialog = wx.FileDialog(self, "Choose dataset file", wildcard=wildcard, 
                              style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        
        if dialog.ShowModal() == wx.ID_OK:
            self.dataset_path = dialog.GetPath()
            self.status_text.AppendText(f"Dataset loaded: {self.dataset_path}\n")
            self.run_item.Enable(True)
            self.SetStatusText(f"Dataset: {os.path.basename(self.dataset_path)}")
        
        dialog.Destroy()
        
    def on_run_analysis(self, event):
        """Handle run analysis"""
        if not hasattr(self, 'dataset_path'):
            wx.MessageBox("Please load a dataset first", "Error", wx.ICON_ERROR)
            return
        
        self.status_text.AppendText("\nStarting analysis...\n")
        self.progress.Pulse()
        self.run_item.Enable(False)
        
        # Run analysis in thread
        thread = AnalysisThread(self, self.dataset_path)
        thread.start()
        
    def update_status(self, message):
        """Update status text"""
        self.status_text.AppendText(f"{message}\n")
        self.progress.Pulse()
        
    def on_analysis_complete(self, analysis):
        """Handle analysis completion"""
        self.analysis = analysis
        self.status_text.AppendText("\nAnalysis complete!\n")
        self.progress.SetValue(100)
        self.results_item.Enable(True)
        self.predict_item.Enable(True)
        self.run_item.Enable(True)
        self.SetStatusText("Analysis complete")
        wx.MessageBox("Analysis completed successfully!", "Success", wx.ICON_INFORMATION)
        
    def on_analysis_error(self, error):
        """Handle analysis error"""
        self.status_text.AppendText(f"\nError: {error}\n")
        self.progress.SetValue(0)
        self.run_item.Enable(True)
        self.SetStatusText("Error occurred")
        wx.MessageBox(f"Error during analysis: {error}", "Error", wx.ICON_ERROR)
        
    def on_view_results(self, event):
        """Handle view results"""
        if self.analysis:
            results_frame = ResultsFrame(self, self.analysis)
            results_frame.Show()
        else:
            wx.MessageBox("Please run analysis first", "Error", wx.ICON_ERROR)
            
    def on_make_prediction(self, event):
        """Handle make prediction"""
        if self.analysis:
            prediction_frame = PredictionFrame(self, self.analysis)
            prediction_frame.Show()
        else:
            wx.MessageBox("Please run analysis first", "Error", wx.ICON_ERROR)
            
    def on_about(self, event):
        """Handle about"""
        info = wx.adv.AboutDialogInfo()
        info.SetName("AI Assistant Usage Behavior Analysis")
        info.SetVersion("1.0")
        info.SetDescription("Comprehensive analysis and prediction tool for AI assistant usage patterns")
        info.SetWebSite("https://github.com")
        info.AddDeveloper("ESCOM - 9th Semester")
        wx.adv.AboutBox(info)
        
    def on_exit(self, event):
        """Handle exit"""
        self.Close(True)


class App(wx.App):
    """Main application"""
    def OnInit(self):
        self.frame = MainFrame()
        self.frame.Show()
        return True


def main():
    """Main entry point"""
    app = App(False)
    app.MainLoop()


if __name__ == '__main__':
    main()
