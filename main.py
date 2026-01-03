"""
AI Assistant Behavior Prediction Agent
Specialized in predicting AI assistant usage behavior patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AIBehaviorPredictionAgent:
    """
    Agente especializado en predicción de comportamiento de asistentes de IA
    Puede predecir múltiples aspectos: satisfacción, tokens usados, duración de sesión, etc.
    """
    
    def __init__(self, dataset_path):
        """
        Inicializa el agente con el dataset
        
        Args:
            dataset_path (str): Ruta al archivo CSV del dataset
        """
        self.dataset_path = dataset_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
    def load_and_preprocess_data(self):
        """Carga y preprocesa el dataset"""
        print("📊 Cargando dataset...")
        self.df = pd.read_csv(self.dataset_path)
        
        # Convertir timestamp a features temporales
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        
        print(f"✅ Dataset cargado: {len(self.df)} registros")
        print(f"\nColumnas: {list(self.df.columns)}")
        print(f"\nEstadísticas básicas:")
        print(self.df.describe())
        
        return self.df
    
    def analyze_data(self):
        """Realiza análisis exploratorio del dataset"""
        print("\n" + "="*60)
        print("🔍 ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*60)
        
        # Distribución por categorías
        print("\n📱 Distribución por Dispositivo:")
        print(self.df['device'].value_counts())
        
        print("\n📚 Distribución por Categoría de Uso:")
        print(self.df['usage_category'].value_counts())
        
        print("\n🤖 Distribución por Modelo de Asistente:")
        print(self.df['assistant_model'].value_counts())
        
        print("\n⭐ Estadísticas de Satisfacción:")
        print(f"Media: {self.df['satisfaction_rating'].mean():.2f}")
        print(f"Mediana: {self.df['satisfaction_rating'].median():.2f}")
        print(f"Desviación estándar: {self.df['satisfaction_rating'].std():.2f}")
        
        print("\n⏱️ Estadísticas de Duración de Sesión (minutos):")
        print(f"Media: {self.df['session_length_minutes'].mean():.2f}")
        print(f"Mediana: {self.df['session_length_minutes'].median():.2f}")
        
        print("\n🎯 Estadísticas de Tokens Usados:")
        print(f"Media: {self.df['tokens_used'].mean():.2f}")
        print(f"Mediana: {self.df['tokens_used'].median():.2f}")
        
    def prepare_features(self, target_column):
        """
        Prepara features para el entrenamiento
        
        Args:
            target_column (str): Columna objetivo a predecir
        """
        # Copiar dataframe
        df_model = self.df.copy()
        
        # Seleccionar features
        categorical_features = ['device', 'usage_category', 'assistant_model']
        numerical_features = ['prompt_length', 'hour', 'day_of_week', 'month']
        
        # Codificar variables categóricas
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_model[col + '_encoded'] = self.label_encoders[col].fit_transform(df_model[col])
            else:
                df_model[col + '_encoded'] = self.label_encoders[col].transform(df_model[col])
        
        # Preparar X e y
        feature_cols = [col + '_encoded' for col in categorical_features] + numerical_features
        
        # Si el target es numérico, añadir más features
        if target_column in ['tokens_used', 'session_length_minutes']:
            feature_cols.append('satisfaction_rating')
        
        X = df_model[feature_cols]
        y = df_model[target_column]
        
        return X, y, feature_cols
    
    def train_satisfaction_predictor(self):
        """Entrena modelo para predecir satisfacción del usuario"""
        print("\n" + "="*60)
        print("🎯 ENTRENANDO PREDICTOR DE SATISFACCIÓN")
        print("="*60)
        
        X, y, feature_cols = self.prepare_features('satisfaction_rating')
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        
        print(f"\n✅ Precisión del modelo: {accuracy:.2%}")
        print("\n📊 Reporte de clasificación:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Importancia de Features:")
        print(importance)
        
        self.models['satisfaction'] = model
        self.feature_importance['satisfaction'] = importance
        
        return model, accuracy
    
    def train_tokens_predictor(self):
        """Entrena modelo para predecir tokens usados"""
        print("\n" + "="*60)
        print("🎯 ENTRENANDO PREDICTOR DE TOKENS USADOS")
        print("="*60)
        
        X, y, feature_cols = self.prepare_features('tokens_used')
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Métricas del modelo:")
        print(f"   MAE: {mae:.2f} tokens")
        print(f"   RMSE: {rmse:.2f} tokens")
        print(f"   R²: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Importancia de Features:")
        print(importance)
        
        self.models['tokens'] = model
        self.feature_importance['tokens'] = importance
        
        return model, r2
    
    def train_session_length_predictor(self):
        """Entrena modelo para predecir duración de sesión"""
        print("\n" + "="*60)
        print("🎯 ENTRENANDO PREDICTOR DE DURACIÓN DE SESIÓN")
        print("="*60)
        
        X, y, feature_cols = self.prepare_features('session_length_minutes')
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluar
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✅ Métricas del modelo:")
        print(f"   MAE: {mae:.2f} minutos")
        print(f"   RMSE: {rmse:.2f} minutos")
        print(f"   R²: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Importancia de Features:")
        print(importance)
        
        self.models['session_length'] = model
        self.feature_importance['session_length'] = importance
        
        return model, r2
    
    def predict_behavior(self, device, usage_category, assistant_model, prompt_length, hour, day_of_week, month):
        """
        Predice el comportamiento completo del asistente de IA
        
        Args:
            device (str): Dispositivo (Desktop, Mobile, Tablet, Smart Speaker)
            usage_category (str): Categoría (Coding, Research, Writing, etc.)
            assistant_model (str): Modelo (GPT-4o, GPT-5, GPT-5.1, o1, Mini)
            prompt_length (int): Longitud del prompt
            hour (int): Hora del día (0-23)
            day_of_week (int): Día de la semana (0=Lunes, 6=Domingo)
            month (int): Mes (1-12)
        """
        print("\n" + "="*60)
        print("🔮 PREDICCIÓN DE COMPORTAMIENTO")
        print("="*60)
        
        # Preparar input
        input_data = {
            'device': device,
            'usage_category': usage_category,
            'assistant_model': assistant_model,
            'prompt_length': prompt_length,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month
        }
        
        print("\n📝 Datos de entrada:")
        for key, value in input_data.items():
            print(f"   {key}: {value}")
        
        # Codificar categóricas
        device_enc = self.label_encoders['device'].transform([device])[0]
        usage_enc = self.label_encoders['usage_category'].transform([usage_category])[0]
        model_enc = self.label_encoders['assistant_model'].transform([assistant_model])[0]
        
        # Features base (sin satisfacción)
        X_base = np.array([[device_enc, usage_enc, model_enc, prompt_length, hour, day_of_week, month]])
        
        # Predecir satisfacción primero
        satisfaction_pred = self.models['satisfaction'].predict(X_base)[0]
        
        # Ahora con satisfacción para tokens y duración
        X_with_satisfaction = np.array([[device_enc, usage_enc, model_enc, prompt_length, hour, day_of_week, month, satisfaction_pred]])
        
        tokens_pred = self.models['tokens'].predict(X_with_satisfaction)[0]
        session_pred = self.models['session_length'].predict(X_with_satisfaction)[0]
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DE LA PREDICCIÓN")
        print("="*60)
        print(f"\n⭐ Satisfacción esperada: {satisfaction_pred}/5")
        print(f"🎯 Tokens a usar: ~{int(tokens_pred)} tokens")
        print(f"⏱️ Duración de sesión: ~{session_pred:.2f} minutos")
        
        # Interpretación
        print("\n💡 Interpretación:")
        if satisfaction_pred >= 4:
            print("   ✅ Alta probabilidad de experiencia positiva")
        elif satisfaction_pred >= 3:
            print("   ⚠️ Experiencia neutral, posible mejora necesaria")
        else:
            print("   ❌ Baja satisfacción esperada, revisar parámetros")
        
        return {
            'satisfaction': satisfaction_pred,
            'tokens': tokens_pred,
            'session_length': session_pred
        }
    
    def generate_insights(self):
        """Genera insights sobre patrones de comportamiento"""
        print("\n" + "="*60)
        print("💡 INSIGHTS Y PATRONES DE COMPORTAMIENTO")
        print("="*60)
        
        # Satisfacción por dispositivo
        print("\n📱 Satisfacción promedio por dispositivo:")
        satisfaction_by_device = self.df.groupby('device')['satisfaction_rating'].mean().sort_values(ascending=False)
        for device, rating in satisfaction_by_device.items():
            print(f"   {device}: {rating:.2f}/5")
        
        # Satisfacción por modelo
        print("\n🤖 Satisfacción promedio por modelo de asistente:")
        satisfaction_by_model = self.df.groupby('assistant_model')['satisfaction_rating'].mean().sort_values(ascending=False)
        for model, rating in satisfaction_by_model.items():
            print(f"   {model}: {rating:.2f}/5")
        
        # Tokens por categoría
        print("\n🎯 Tokens promedio por categoría:")
        tokens_by_category = self.df.groupby('usage_category')['tokens_used'].mean().sort_values(ascending=False)
        for category, tokens in tokens_by_category.items():
            print(f"   {category}: {tokens:.0f} tokens")
        
        # Hora más activa
        print("\n⏰ Horas con mayor actividad:")
        hour_activity = self.df.groupby('hour').size().sort_values(ascending=False).head(5)
        for hour, count in hour_activity.items():
            print(f"   {hour}:00 - {count} sesiones")
        
        # Correlaciones
        print("\n🔗 Correlaciones importantes:")
        corr = self.df[['prompt_length', 'session_length_minutes', 'satisfaction_rating', 'tokens_used']].corr()
        print(corr)


def main():
    """Función principal"""
    print("="*60)
    print("🤖 AI ASSISTANT BEHAVIOR PREDICTION AGENT")
    print("="*60)
    
    # Inicializar agente
    agent = AIBehaviorPredictionAgent('Daily_AI_Assistant_Usage_Behavior_Dataset.csv')
    
    # Cargar y analizar datos
    agent.load_and_preprocess_data()
    agent.analyze_data()
    
    # Entrenar modelos
    agent.train_satisfaction_predictor()
    agent.train_tokens_predictor()
    agent.train_session_length_predictor()
    
    # Generar insights
    agent.generate_insights()
    
    # Ejemplos de predicción
    print("\n" + "="*60)
    print("🧪 EJEMPLOS DE PREDICCIÓN")
    print("="*60)
    
    # Ejemplo 1: Coding en Desktop con GPT-5
    print("\n--- Ejemplo 1: Sesión de Coding ---")
    agent.predict_behavior(
        device='Desktop',
        usage_category='Coding',
        assistant_model='GPT-5',
        prompt_length=150,
        hour=14,
        day_of_week=2,  # Miércoles
        month=1
    )
    
    # Ejemplo 2: Research en Mobile con o1
    print("\n--- Ejemplo 2: Sesión de Research ---")
    agent.predict_behavior(
        device='Mobile',
        usage_category='Research',
        assistant_model='o1',
        prompt_length=80,
        hour=20,
        day_of_week=5,  # Sábado
        month=1
    )
    
    # Ejemplo 3: Entertainment en Tablet
    print("\n--- Ejemplo 3: Sesión de Entertainment ---")
    agent.predict_behavior(
        device='Tablet',
        usage_category='Entertainment',
        assistant_model='Mini',
        prompt_length=50,
        hour=22,
        day_of_week=6,  # Domingo
        month=1
    )
    
    print("\n" + "="*60)
    print("✅ AGENTE COMPLETAMENTE OPERATIVO")
    print("="*60)


if __name__ == '__main__':
    main()
