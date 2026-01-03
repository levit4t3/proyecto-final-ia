# 🤖 AI Assistant Behavior Prediction Agent

Agente de IA especializado en predecir el comportamiento de asistentes de IA basado en patrones de uso históricos.

## 📋 Descripción

Este agente utiliza Machine Learning para predecir múltiples aspectos del comportamiento de asistentes de IA:
- ⭐ **Satisfacción del usuario** (1-5)
- 🎯 **Tokens consumidos** en una sesión
- ⏱️ **Duración de la sesión** en minutos

## 🎯 Características

### Modelos de Predicción
1. **Predictor de Satisfacción**: RandomForestClassifier para clasificar niveles de satisfacción
2. **Predictor de Tokens**: GradientBoostingRegressor para estimar consumo de tokens
3. **Predictor de Duración**: RandomForestRegressor para estimar duración de sesión

### Análisis Exploratorio
- Distribución por dispositivo, categoría y modelo de asistente
- Estadísticas de satisfacción, tokens y duración de sesión
- Identificación de patrones temporales (hora, día, mes)
- Correlaciones entre variables

### Insights Generados
- Satisfacción promedio por dispositivo y modelo
- Consumo de tokens por categoría de uso
- Horas de mayor actividad
- Correlaciones importantes entre variables

## 🚀 Uso

### Ejecutar el Agente Completo

```bash
python main.py
```

Esto ejecutará:
1. Carga y análisis del dataset
2. Entrenamiento de los 3 modelos predictivos
3. Generación de insights
4. Ejemplos de predicción

### Usar el Agente Programáticamente

```python
from main import AIBehaviorPredictionAgent

# Inicializar agente
agent = AIBehaviorPredictionAgent('Daily_AI_Assistant_Usage_Behavior_Dataset.csv')

# Cargar datos
agent.load_and_preprocess_data()

# Entrenar modelos
agent.train_satisfaction_predictor()
agent.train_tokens_predictor()
agent.train_session_length_predictor()

# Hacer una predicción
prediction = agent.predict_behavior(
    device='Desktop',
    usage_category='Coding',
    assistant_model='GPT-5',
    prompt_length=150,
    hour=14,
    day_of_week=2,
    month=1
)

print(f"Satisfacción: {prediction['satisfaction']}")
print(f"Tokens: {prediction['tokens']}")
print(f"Duración: {prediction['session_length']} minutos")
```

## 📊 Dataset

El dataset incluye las siguientes características:
- **timestamp**: Fecha y hora de la sesión
- **device**: Dispositivo usado (Desktop, Mobile, Tablet, Smart Speaker)
- **usage_category**: Categoría de uso (Coding, Research, Writing, Education, etc.)
- **prompt_length**: Longitud del prompt
- **session_length_minutes**: Duración de la sesión
- **satisfaction_rating**: Calificación de satisfacción (1-5)
- **assistant_model**: Modelo de asistente (GPT-4o, GPT-5, GPT-5.1, o1, Mini)
- **tokens_used**: Cantidad de tokens consumidos

## 🛠️ Dependencias

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Instalar con:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 📈 Métricas de Rendimiento

El agente reporta las siguientes métricas:

### Predictor de Satisfacción
- Precisión (Accuracy)
- Reporte de clasificación completo
- Importancia de features

### Predictores de Tokens y Duración
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)
- Importancia de features

## 🎓 Casos de Uso

1. **Optimización de Experiencia**: Predecir satisfacción antes de una interacción
2. **Gestión de Recursos**: Estimar consumo de tokens para planificación
3. **Análisis de Patrones**: Identificar tendencias de uso
4. **Recomendaciones**: Sugerir mejores configuraciones basadas en predicciones

## 🔮 Ejemplos de Predicción

El agente incluye ejemplos predefinidos:
- Sesión de Coding en Desktop
- Sesión de Research en Mobile
- Sesión de Entertainment en Tablet

## 📝 Notas

- Los modelos se entrenan automáticamente al ejecutar el script
- Las predicciones son probabilísticas basadas en patrones históricos
- La precisión mejora con más datos de entrenamiento

## 👨‍💻 Autor

Proyecto Final - Inteligencia Artificial
ESCOM, 9° Semestre
