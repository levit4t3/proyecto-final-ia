import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from analysis import AIUsageBehaviorAnalysis


class AnalysisThread(threading.Thread):
    """Thread for running analysis in background"""
    def __init__(self, parent, dataset_path):
        threading.Thread.__init__(self)
        self.parent = parent
        self.dataset_path = dataset_path
        self.analysis = None
        self.daemon = True
        
    def run(self):
        """Ejecutar análisis"""
        try:
            self.parent.update_status("Cargando dataset...")
            self.analysis = AIUsageBehaviorAnalysis(self.dataset_path)
            self.analysis.load_data()
            
            self.parent.update_status("Ejecutando análisis exploratorio...")
            self.analysis.basic_statistics()
            self.analysis.exploratory_data_analysis()
            
            self.parent.update_status("Entrenando modelo de satisfacción...")
            self.analysis.train_satisfaction_model()
            
            self.parent.update_status("Entrenando modelo de tokens...")
            self.analysis.train_tokens_model()
            
            self.parent.update_status("Entrenando modelo de duración de sesión...")
            self.analysis.train_session_length_model()
            
            self.parent.update_status("Generando insights...")
            self.analysis.generate_insights()
            
            self.parent.on_analysis_complete(self.analysis)
        except Exception as e:
            self.parent.on_analysis_error(str(e))


class ResultsWindow(tk.Toplevel):
    """Ventana para mostrar resultados del análisis"""
    def __init__(self, parent, analysis):
        super().__init__(parent)
        self.analysis = analysis
        self.title("Resultados del Análisis")
        self.geometry("960x700")
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Estadísticas")
        self.create_statistics_panel(stats_frame)
        
        # Model Performance tab
        models_frame = ttk.Frame(notebook)
        notebook.add(models_frame, text="Rendimiento de Modelos")
        self.create_models_panel(models_frame)
        
        # Insights tab
        insights_frame = ttk.Frame(notebook)
        notebook.add(insights_frame, text="Insights")
        self.create_insights_panel(insights_frame)
        
    def create_statistics_panel(self, parent):
        """Create statistics panel"""
        text_widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=('Courier', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_str = f"Forma del Dataset: {self.analysis.df.shape[0]} filas x {self.analysis.df.shape[1]} columnas\n\n"
        info_str += "Estadísticas Descriptivas:\n"
        info_str += str(self.analysis.df.describe()) + "\n\n"
        info_str += "Variables Categóricas:\n"
        for col in ['device', 'usage_category', 'assistant_model']:
            info_str += f"\n{col}:\n{self.analysis.df[col].value_counts()}\n"
        
        text_widget.insert(tk.END, info_str)
        text_widget.config(state=tk.DISABLED)
        
    def create_models_panel(self, parent):
        """Create model performance panel"""
        text_widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=('Courier', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_str = "="*70 + "\n"
        info_str += "Modelo de Satisfacción (Random Forest Classifier)\n"
        info_str += "="*70 + "\n"
        info_str += f"Precisión: {self.analysis.results['satisfaction']['accuracy']:.4f}\n\n"
        info_str += "Características Principales:\n"
        info_str += str(self.analysis.results['satisfaction']['feature_importance'].head()) + "\n\n"
        
        info_str += "="*70 + "\n"
        info_str += "Modelo de Tokens (Gradient Boosting Regressor)\n"
        info_str += "="*70 + "\n"
        info_str += f"MAE: {self.analysis.results['tokens']['mae']:.2f}\n"
        info_str += f"RMSE: {self.analysis.results['tokens']['rmse']:.2f}\n"
        info_str += f"R2: {self.analysis.results['tokens']['r2']:.4f}\n\n"
        info_str += "Características Principales:\n"
        info_str += str(self.analysis.results['tokens']['feature_importance'].head()) + "\n\n"
        
        info_str += "="*70 + "\n"
        info_str += "Modelo de Duración de Sesión (Random Forest Regressor)\n"
        info_str += "="*70 + "\n"
        info_str += f"MAE: {self.analysis.results['session_length']['mae']:.2f}\n"
        info_str += f"RMSE: {self.analysis.results['session_length']['rmse']:.2f}\n"
        info_str += f"R2: {self.analysis.results['session_length']['r2']:.4f}\n\n"
        info_str += "Características Principales:\n"
        info_str += str(self.analysis.results['session_length']['feature_importance'].head())
        
        text_widget.insert(tk.END, info_str)
        text_widget.config(state=tk.DISABLED)
        
    def create_insights_panel(self, parent):
        """Create insights panel"""
        text_widget = scrolledtext.ScrolledText(parent, wrap=tk.WORD, font=('Courier', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        insights_str = "INSIGHTS CLAVE\n" + "="*70 + "\n\n"
        
        insights_str += "1. Patrones de Uso por Dispositivo:\n"
        insights_str += str(self.analysis.df.groupby('device').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "2. Rendimiento por Modelo:\n"
        insights_str += str(self.analysis.df.groupby('assistant_model').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "3. Análisis por Categoría de Uso:\n"
        insights_str += str(self.analysis.df.groupby('usage_category').agg({
            'satisfaction_rating': 'mean',
            'tokens_used': 'mean',
            'session_length_minutes': 'mean'
        }).round(2)) + "\n\n"
        
        insights_str += "4. Patrones Temporales:\n"
        insights_str += "Horas más activas:\n"
        insights_str += str(self.analysis.df.groupby('hour').size().sort_values(ascending=False).head())
        
        text_widget.insert(tk.END, insights_str)
        text_widget.config(state=tk.DISABLED)


class PredictionWindow(tk.Toplevel):
    """Ventana para hacer predicciones"""
    def __init__(self, parent, analysis):
        super().__init__(parent)
        self.analysis = analysis
        self.title("Hacer Predicción")
        self.geometry("600x550")
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input form
        form_frame = ttk.LabelFrame(main_frame, text="Parámetros de Entrada", padding="10")
        form_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Device
        ttk.Label(form_frame, text="Dispositivo:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar(value='Desktop')
        device_combo = ttk.Combobox(form_frame, textvariable=self.device_var, 
                                    values=['Desktop', 'Mobile', 'Tablet', 'Smart Speaker'],
                                    state='readonly', width=30)
        device_combo.grid(row=0, column=1, pady=5, padx=5)
        
        # Usage Category
        ttk.Label(form_frame, text="Categoría de Uso:").grid(row=1, column=0, sticky=tk.W, pady=5)
        categories = list(self.analysis.df['usage_category'].unique())
        self.category_var = tk.StringVar(value=categories[0])
        category_combo = ttk.Combobox(form_frame, textvariable=self.category_var,
                                      values=categories, state='readonly', width=30)
        category_combo.grid(row=1, column=1, pady=5, padx=5)
        
        # Assistant Model
        ttk.Label(form_frame, text="Modelo de Asistente:").grid(row=2, column=0, sticky=tk.W, pady=5)
        models = list(self.analysis.df['assistant_model'].unique())
        self.model_var = tk.StringVar(value=models[0])
        model_combo = ttk.Combobox(form_frame, textvariable=self.model_var,
                                   values=models, state='readonly', width=30)
        model_combo.grid(row=2, column=1, pady=5, padx=5)
        
        # Prompt Length
        ttk.Label(form_frame, text="Longitud del Prompt:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.prompt_var = tk.IntVar(value=100)
        prompt_spin = ttk.Spinbox(form_frame, from_=1, to=500, textvariable=self.prompt_var, width=30)
        prompt_spin.grid(row=3, column=1, pady=5, padx=5)
        
        # Hour
        ttk.Label(form_frame, text="Hora (0-23):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.hour_var = tk.IntVar(value=12)
        hour_spin = ttk.Spinbox(form_frame, from_=0, to=23, textvariable=self.hour_var, width=30)
        hour_spin.grid(row=4, column=1, pady=5, padx=5)
        
        # Day of Week
        ttk.Label(form_frame, text="Día de la Semana:").grid(row=5, column=0, sticky=tk.W, pady=5)
        days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        self.day_var = tk.StringVar(value='Lunes')
        day_combo = ttk.Combobox(form_frame, textvariable=self.day_var,
                                values=days, state='readonly', width=30)
        day_combo.grid(row=5, column=1, pady=5, padx=5)
        
        # Month
        ttk.Label(form_frame, text="Mes (1-12):").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.month_var = tk.IntVar(value=1)
        month_spin = ttk.Spinbox(form_frame, from_=1, to=12, textvariable=self.month_var, width=30)
        month_spin.grid(row=6, column=1, pady=5, padx=5)
        
        # Predict button
        predict_btn = ttk.Button(main_frame, text="Predecir", command=self.on_predict)
        predict_btn.pack(pady=10)
        
        # Results
        results_frame = ttk.LabelFrame(main_frame, text="Resultados de la Predicción", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                      font=('Courier', 10, 'bold'),
                                                      height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
    def on_predict(self):
        """Handle predict button click"""
        try:
            device = self.device_var.get()
            category = self.category_var.get()
            model = self.model_var.get()
            prompt_length = self.prompt_var.get()
            hour = self.hour_var.get()
            day_of_week = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 
                          'Viernes', 'Sábado', 'Domingo'].index(self.day_var.get())
            month = self.month_var.get()
            
            result = self.analysis.predict(device, category, model, prompt_length, 
                                          hour, day_of_week, month)
            
            result_str = "RESULTADOS DE LA PREDICCIÓN\n" + "="*50 + "\n\n"
            result_str += f"Calificación de Satisfacción: {result['satisfaction']}/5\n"
            result_str += f"Tokens Usados: {result['tokens']} tokens\n"
            result_str += f"Duración de Sesión: {result['session_length']} minutos\n\n"
            
            if result['satisfaction'] >= 4:
                result_str += "Interpretación: Se espera alta satisfacción\n"
            elif result['satisfaction'] >= 3:
                result_str += "Interpretación: Se espera experiencia neutral\n"
            else:
                result_str += "Interpretación: Baja satisfacción - revisar parámetros\n"
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_str)
            self.results_text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al hacer la predicción: {str(e)}")


class MainApplication(tk.Tk):
    """Ventana principal de la aplicación"""
    def __init__(self):
        super().__init__()
        self.analysis = None
        # Auto-load dataset from current directory
        self.dataset_path = "Daily_AI_Assistant_Usage_Behavior_Dataset.csv"
        self.title("Análisis de Comportamiento de Asistentes de IA")
        self.geometry("1920x1080")
        self.init_ui()
        self.check_dataset()
        
    def init_ui(self):
        """Initialize UI"""
        # Menu bar (simplified)
        menubar = tk.Menu(self)
        self.config(menu=menubar)
                
        # Analysis menu
        self.analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Análisis", menu=self.analysis_menu)
        self.analysis_menu.add_command(label="Ver Resultados", command=self.on_view_results,
                                      accelerator="Ctrl+V", state=tk.DISABLED)
        self.analysis_menu.add_command(label="Hacer Predicción", command=self.on_make_prediction,
                                      accelerator="Ctrl+P", state=tk.DISABLED)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ayuda", menu=help_menu)
        help_menu.add_command(label="Acerca de", command=self.on_about)
        
        # Keyboard shortcuts
        self.bind('<Control-v>', lambda e: self.on_view_results())
        self.bind('<Control-p>', lambda e: self.on_make_prediction())
        self.bind('<Control-q>', lambda e: self.quit())
        
        # Main content
        main_frame = ttk.Frame(self, padding="30")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title with icon/styling
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=(0, 20))
        
        title_label = tk.Label(title_frame, 
                              text="Análisis de Comportamiento de Asistentes de IA",
                              font=('TkDefaultFont', 18, 'bold'),
                              fg='#2c3e50')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Sistema de Predicción con Machine Learning",
                                 font=('TkDefaultFont', 10),
                                 fg='#7f8c8d')
        subtitle_label.pack()
        
        # Dataset info
        dataset_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="15")
        dataset_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.dataset_info_label = tk.Label(dataset_frame, 
                                          text="Dataset: Daily_AI_Assistant_Usage_Behavior_Dataset.csv\nEstado: Listo para analizar",
                                          font=('TkDefaultFont', 9),
                                          justify=tk.LEFT,
                                          fg='#27ae60')
        self.dataset_info_label.pack()
        
        # Instructions
        instructions_frame = ttk.LabelFrame(main_frame, text="Qué sucederá", padding="15")
        instructions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        instructions = (
            "Al hacer clic en INICIAR ANÁLISIS, el sistema:\n\n"
            "1. Cargará y analizará el dataset (300 registros)\n"
            "2. Realizará análisis exploratorio de datos\n"
            "3. Entrenará 3 modelos de machine learning:\n"
            "   - Predictor de satisfacción (Random Forest)\n"
            "   - Predictor de uso de tokens (Gradient Boosting)\n"
            "   - Predictor de duración de sesión (Random Forest)\n"
            "4. Generará 11 imágenes de visualización (archivos PNG)\n"
            "5. Calculará importancia de características e insights\n\n"
            "Al finalizar, podrás:\n"
            "- Ver resultados detallados en pestañas organizadas\n"
            "- Hacer predicciones para nuevos escenarios\n"
            "- Revisar los archivos PNG generados en la carpeta del proyecto"
        )
        instructions_text = tk.Label(instructions_frame, 
                                    text=instructions, 
                                    justify=tk.LEFT,
                                    font=('TkDefaultFont', 9))
        instructions_text.pack(anchor=tk.W)
        
        # Big centered START ANALYSIS button
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=30)
        
        self.start_button = tk.Button(button_frame,
                                     text="INICIAR ANÁLISIS",
                                     command=self.on_run_analysis,
                                     font=('TkDefaultFont', 16, 'bold'),
                                     bg='#3498db',
                                     fg='white',
                                     activebackground='#2980b9',
                                     activeforeground='white',
                                     padx=50,
                                     pady=20,
                                     cursor='hand2',
                                     relief=tk.RAISED,
                                     borderwidth=3)
        self.start_button.pack()
        
        # Status and progress section
        status_frame = ttk.LabelFrame(main_frame, text="Estado del Análisis", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, 
                                                     font=('Courier', 9), height=8)
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.status_text.insert(tk.END, "Listo para iniciar el análisis.\n")
        self.status_text.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X)
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Listo - Dataset cargado", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def check_dataset(self):
        """Verificar si el dataset existe"""
        if not os.path.exists(self.dataset_path):
            self.dataset_info_label.config(
                text=f"ERROR: ¡Archivo del dataset no encontrado!\nAsegúrate de que '{self.dataset_path}' existe en el directorio actual.",
                fg='#e74c3c'
            )
            self.start_button.config(state=tk.DISABLED, bg='#95a5a6')
            self.status_bar.config(text="Error: Dataset no encontrado")
            messagebox.showerror("Dataset No Encontrado", 
                               f"No se encontró el archivo '{self.dataset_path}'.\n"
                               "Asegúrate de que existe en el mismo directorio que este programa.")
        else:
            self.update_status_text(f"Dataset encontrado: {self.dataset_path}\n")
            self.status_bar.config(text=f"Listo - Dataset: {self.dataset_path}")
    
    def on_run_analysis(self):
        """Manejar ejecución del análisis"""
        self.update_status_text("\nIniciando análisis...\n")
        self.update_status_text("Esto puede tomar 30-60 segundos...\n\n")
        self.progress.start(10)
        self.start_button.config(state=tk.DISABLED, bg='#95a5a6', text="ANALIZANDO...")
        
        # Run analysis in thread
        thread = AnalysisThread(self, self.dataset_path)
        thread.start()
        
    def update_status(self, message):
        """Update status text (called from thread)"""
        self.after(0, self._update_status_impl, message)
        
    def _update_status_impl(self, message):
        """Implementation of status update"""
        self.update_status_text(f"{message}\n")
        
    def update_status_text(self, message):
        """Update status text widget"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        
    def on_analysis_complete(self, analysis):
        """Manejar finalización del análisis"""
        self.analysis = analysis
        self.update_status_text("\n¡Análisis completado!\n")
        self.update_status_text("Todas las visualizaciones PNG guardadas en la carpeta del proyecto.\n")
        self.progress.stop()
        self.start_button.config(state=tk.NORMAL, bg='#27ae60', text="ANÁLISIS COMPLETADO ✓",
                                fg='white', activebackground='#229954')
        self.analysis_menu.entryconfig(0, state=tk.NORMAL)
        self.analysis_menu.entryconfig(1, state=tk.NORMAL)
        self.status_bar.config(text="Análisis completado - Imágenes guardadas")
        
        # Show completion message with image info
        message = (
            "¡Análisis completado exitosamente!\n\n"
            "Visualizaciones generadas guardadas como archivos PNG:\n"
            "1. numerical_distributions.png\n"
            "2. categorical_distributions.png\n"
            "3. correlation_matrix.png\n"
            "4. satisfaction_by_device.png\n"
            "5. hourly_usage.png\n"
            "6. tokens_by_category.png\n"
            "7. session_by_model.png\n"
            "8. confusion_matrix_satisfaction.png\n"
            "9. feature_importance_satisfaction.png\n"
            "10. tokens_prediction.png\n"
            "11. session_length_prediction.png\n\n"
            "Ahora puedes:\n"
            "- Ver Resultados (menú Análisis)\n"
            "- Hacer Predicciones (menú Análisis)"
        )
        messagebox.showinfo("Éxito", message)
        
    def on_analysis_error(self, error):
        """Manejar error en el análisis"""
        self.update_status_text(f"\nError: {error}\n")
        self.progress.stop()
        self.start_button.config(state=tk.NORMAL, bg='#e74c3c', text="ERROR - INTENTAR DE NUEVO",
                                activebackground='#c0392b')
        self.status_bar.config(text="Ocurrió un error")
        messagebox.showerror("Error", f"Error durante el análisis: {error}")
        
    def on_view_results(self):
        """Manejar ver resultados"""
        if self.analysis:
            ResultsWindow(self, self.analysis)
        else:
            messagebox.showerror("Error", "Por favor ejecuta el análisis primero")
            
    def on_make_prediction(self):
        """Manejar hacer predicción"""
        if self.analysis:
            PredictionWindow(self, self.analysis)
        else:
            messagebox.showerror("Error", "Por favor ejecuta el análisis primero")
        
    def on_about(self):
        """Manejar acerca de"""
        about_text = (
            "Análisis de Comportamiento de Asistentes de IA\n"
            "Versión 1.0\n\n"
            "Herramienta completa de análisis y predicción para\n"
            "patrones de uso de asistentes de IA\n\n"
            "Dataset: Daily_AI_Assistant_Usage_Behavior_Dataset.csv\n"
            "Modelos: Random Forest, Gradient Boosting\n"
            "Visualizaciones: 11 archivos PNG\n\n"
            "ESCOM - 9° Semestre"
        )
        messagebox.showinfo("Acerca de", about_text)


def main():
    """Main entry point"""
    app = MainApplication()
    app.mainloop()


if __name__ == '__main__':
    main()
