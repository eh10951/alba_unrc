# 🎓 Asistente Escolar Alba - UNRC

## 📋 Descripción General

Alba es un asistente virtual inteligente diseñado para estudiantes universitarios de la Universidad Nacional de Río Cuarto (UNRC). Utiliza **machine learning** y **reconocimiento de voz** para clasificar consultas académicas y proporcionar consejos personalizados.

## 🧠 Arquitectura del Modelo de Machine Learning

### � **Paso 1: Importación de Librerías**

```python
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS
import os
import random
```

**¿Qué hace cada librería?**
- `Flask`: Framework web para crear el servidor HTTP
- `CountVectorizer`: Convierte texto a vectores numéricos (Bag of Words)
- `LogisticRegression`: Algoritmo de clasificación supervisada
- `flask_cors`: Permite peticiones desde diferentes dominios (CORS)
- `os`, `random`: Utilidades del sistema y números aleatorios

### 🏗️ **Paso 2: Configuración del Servidor Flask**

```python
app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')


    <!-- 
    Esta función de Flask sirve archivos estáticos desde el directorio '../frontend' cuando se accede a la ruta '/<path:filename>'. 
    Por ejemplo, permite servir archivos como imágenes, hojas de estilo o scripts al frontend de la aplicación.
    -->

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

```

**Propósito:**
- Crea la aplicación web
- Habilita CORS para comunicación frontend-backend
- Sirve los archivos estáticos (HTML, CSS, imágenes)

### 📝 **Paso 3: Creación del Dataset de Entrenamiento**

```python
# Entrenamiento simple con datos básicos
vectorizer = CountVectorizer()
```
<!-- 
Este bloque muestra cómo se inicia el proceso de vectorización del texto usando CountVectorizer. 
CountVectorizer convierte los textos de ejemplo en vectores numéricos para que el modelo de machine learning pueda procesarlos.
Es un paso esencial para transformar datos textuales en una representación que el algoritmo pueda entender y utilizar para el entrenamiento.
-->

El modelo se entrena con **8 categorías** diferentes:

#### 🔢 **Categoría: Matemáticas**
```python
textos_matematicas = [
    "necesito ayuda con álgebra", "no entiendo las ecuaciones", "me cuesta geometría",
    "ayuda con cálculo", "problemas con matemáticas", "dificultades en estadística"
]
```
**Función:** Identifica consultas relacionadas con matemáticas, álgebra, cálculo, etc.

#### ⚡ **Categoría: Física**
```python
textos_fisica = [
    "no entiendo la mecánica", "problemas con fuerzas", "cinemática es difícil",
    "ayuda con física", "problemas de física", "no entiendo física"
]
```
**Función:** Detecta preguntas sobre mecánica, fuerzas, cinemática, etc.

#### 🧪 **Categoría: Química**
```python
textos_quimica = [
    "tabla periódica", "enlaces químicos", "reacciones químicas",
    "ayuda con química", "problemas de química", "no entiendo química"
]
```
**Función:** Reconoce consultas sobre elementos, reacciones, enlaces químicos.

#### � **Categoría: Programación**
```python
textos_programacion = [
    "aprender a programar", "algoritmos difíciles", "estructuras de datos",
    "ayuda con programación", "problemas de código", "no entiendo programación"
]
```
**Función:** Identifica consultas sobre código, algoritmos, desarrollo de software.

#### ⚠️ **Categoría: Deserción Académica (CRÍTICA)**
```python
textos_desercion = [
    "quiero dejar la escuela", "no tengo motivación para nada", "me siento muy cansado de estudiar",
    "ya no quiero ir a clases nunca", "voy a abandonar mis estudios", "no me sirve estudiar esto",
    # ... 27 ejemplos más
]
```
**Propósito CRÍTICO:** Detecta estudiantes en riesgo de abandonar sus estudios para intervención temprana.

#### 🚀 **Categoría: Motivación**
```python
textos_motivacion = [
    "busco motivación para estudiar", "cómo ser mejor estudiante", "técnicas de estudio efectivas",
    "quiero mejorar mis hábitos de estudio", "necesito consejos para estudiar mejor",
    # ... más ejemplos

]
```
**Función:** Identifica estudiantes que buscan mejorar su rendimiento académico.

#### � **Categoría: Comentarios Positivos**
```python
textos_positivos_escuela = [
    "me gusta la escuela", "me gusta estudiar mucho", "me gusta la universidad",
    "me encanta aprender", "disfruto ir a clases", "me gusta mi carrera",
    # ... 30+ ejemplos más
]
```
**Función:** Reconoce estudiantes con actitud positiva hacia la educación.

#### 😰 **Categoría: Comentarios Negativos**
```python
textos_negativos = [
    "esta materia es muy difícil", "no me gusta esta clase", "el profesor explica mal",
    "estoy reprobando materias", "voy mal en mis calificaciones", "saqué malas notas",
    # ... más ejemplos
]
```
**Función:** Detecta frustración académica sin intención de abandono.

--------------------------------------------------------- 
### 🔄 **Paso 4: Preparación de Datos para Entrenamiento**

```python

¿Qué significa ["matematicas"] * len(textos_matematicas)?
["matematicas"] crea una lista con un solo elemento: "matematicas".
len(textos_matematicas) obtiene la cantidad de elementos en la lista textos_matematicas.
Al multiplicar la lista por un número, por ejemplo ["matematicas"] * 3, el resultado es ["matematicas", "matematicas", "matematicas"].
¿Por qué se usa así?
Esta técnica se utiliza para crear una lista de etiquetas (labels) que tenga la misma longitud que la lista de textos de esa categoría. Por ejemplo, si tienes 5 textos de matemáticas, quieres una lista con 5 veces la etiqueta "matematicas"

-------------------------------------------------------------
# Combinar todos los textos
todos_los_textos = (textos_matematicas + textos_fisica + textos_quimica + 
                   textos_programacion + textos_desercion + textos_motivacion + 
                   textos_positivos_escuela + textos_negativos)

# Crear etiquetas correspondientes
todas_las_etiquetas = (["matematicas"] * len(textos_matematicas) +
                      ["fisica"] * len(textos_fisica) +
                      # ... para cada categoría
                      ["negativo"] * len(textos_negativos))
```

**¿Qué hace esto?**
- Combina todos los ejemplos de texto en una lista única
- Crea etiquetas (labels) correspondientes para cada texto
- Prepara el dataset en formato supervised learning (X, y)

### 🎯 **Paso 5: Vectorización del Texto**

```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas
```

**Proceso de CountVectorizer:**
1. **Tokenización**: Separa el texto en palabras individuales
2. **Creación de vocabulario**: Identifica todas las palabras únicas
3. **Vectorización**: Convierte cada texto en un vector numérico

**Ejemplo:**
- Texto: "ayuda con matemáticas"
- Vector: [0, 1, 0, 1, 0, 0, 1, 0, ...] (1 donde aparece la palabra, 0 donde no)

### 🤖 **Paso 6: Entrenamiento del Modelo**

```python
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X, y)
```

**Parámetros de LogisticRegression:**
- `max_iter=1000`: Máximo 1000 iteraciones para convergencia
- `random_state=42`: Semilla para reproducibilidad
- `C=1.0`: Parámetro de regularización (controla overfitting)

**¿Por qué Logistic Regression?**
- Rápido para entrenar y predecir
- Funciona bien con datos de texto
- Proporciona probabilidades de predicción
- Interpretable y estable

### 🎯 **Paso 7: Sistema de Clasificación Inteligente**

```python
@app.route("/clasificar", methods=["POST"])
def clasificar():
    # ... código de manejo de errores
    
    # 1. Clasificación por ML
    X_test = vectorizer.transform([texto])
    pred = model.predict(X_test)[0]
    probabilidades = model.predict_proba(X_test)[0]
    confianza = max(probabilidades)
```

**Proceso de Predicción:**
1. Transforma el texto nuevo usando el mismo vectorizer
2. Obtiene predicción del modelo entrenado
3. Calcula confianza de la predicción

### 🔍 **Paso 8: Sistema de Corrección Inteligente**

```python
# PRIORIDAD 1: Frases explícitamente NEGATIVAS
if any(frase in texto for frase in frases_muy_negativas):
    pred = "negativo"
    confianza = 0.98

# PRIORIDAD 2: Verificar deserción
elif any(palabra in texto for palabra in palabras_desercion):
    pred = "desercion"
    confianza = 0.95

# PRIORIDAD 3: Verificar frases positivas
elif any(frase in texto for frase in frases_muy_positivas):
    pred = "positivo"
    confianza = 0.93
```

**¿Por qué este sistema?**
- **Corrección post-modelo**: El ML puede fallar, este sistema corrige
- **Prioridades claras**: Casos críticos (deserción) tienen máxima prioridad
- **Detección de patrones**: Busca frases específicas no capturadas por el modelo

### 💡 **Paso 9: Sistema de Consejos Profesionales**

```python
consejos = {
    "matematicas": [
        "CURSOS RECOMENDADOS: Inscríbete en 'Fundamentos de Álgebra' de Khan Academy...",
        "APOYO PERSONALIZADO: Solicita tutoría con estudiantes de Ingeniería...",
        # 5 consejos específicos por categoría
    ],
    # ... consejos para cada categoría
}
```

**Características de los consejos:**
- **Específicos**: Recursos reales de la universidad
- **Accionables**: Pasos concretos que el estudiante puede seguir
- **Profesionales**: Incluyen contactos, horarios, ubicaciones
- **Variados**: Se elige aleatoriamente para evitar repetición

### 📊 **Paso 10: Respuesta Estructurada**

```python
return jsonify({
    "respuesta": respuesta,
    "consejo": consejo,
    "categoria": pred,
    "confianza": f"{confianza:.0%}"
})
```

**Salida del sistema:**
- **Respuesta**: Mensaje natural de reconocimiento
- **Consejo**: Recomendación específica profesional
- **Categoría**: Clasificación técnica
- **Confianza**: Porcentaje de certeza del modelo

## 🔧 **Instalación y Ejecución**

### Requisitos
```bash
pip install flask flask-cors scikit-learn numpy
```

### Ejecutar
```bash
cd backend
python app.py
```

### Probar
```bash
curl -X POST http://localhost:5000/clasificar \
  -H "Content-Type: application/json" \
  -d '{"texto": "me gusta estudiar matemáticas"}'
```

## � **Métricas del Modelo**

- **Total de ejemplos de entrenamiento**: ~200+ frases
- **Categorías**: 8 clases diferentes
- **Algoritmo**: Regresión Logística
- **Vectorización**: Bag of Words (CountVectorizer)
- **Sistema de corrección**: Basado en reglas + ML híbrido

## � **Casos de Uso Principales**

1. **Detección de riesgo académico**: Identifica estudiantes que podrían abandonar
2. **Clasificación de consultas**: Dirige automáticamente a recursos específicos
3. **Apoyo motivacional**: Reconoce estudiantes que necesitan aliento
4. **Tutoría inteligente**: Sugiere recursos según la materia consultada

---
*Desarrollado para la Universidad Nacional de Río Cuarto (UNRC) | 2025*