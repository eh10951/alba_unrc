from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS
import os
import random

app = Flask(__name__)
CORS(app)

# Configuración para servir archivos estáticos
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

# Entrenamiento simple con datos básicos
vectorizer = CountVectorizer()

# Ejemplos básicos de categorías académicas
textos_matematicas = [
    "necesito ayuda con álgebra", "no entiendo las ecuaciones", "me cuesta geometría",
    "ayuda con cálculo", "problemas con matemáticas", "dificultades en estadística"
]

textos_fisica = [
    "no entiendo la mecánica", "problemas con fuerzas", "cinemática es difícil",
    "ayuda con física", "problemas de física", "no entiendo física"
]

textos_quimica = [
    "tabla periódica", "enlaces químicos", "reacciones químicas",
    "ayuda con química", "problemas de química", "no entiendo química"
]

textos_programacion = [
    "aprender a programar", "algoritmos difíciles", "estructuras de datos",
    "ayuda con programación", "problemas de código", "no entiendo programación"
]

textos_desercion = [
    # Frases CLARAMENTE negativas de abandono
    "quiero dejar la escuela", "no tengo motivación para nada", "me siento muy cansado de estudiar",
    "ya no quiero ir a clases nunca", "voy a abandonar mis estudios", "no me sirve estudiar esto",
    "odio ir a la universidad", "no aguanto más", "me quiero salir definitivamente",
    "esto no es para mí", "quiero renunciar", "no puedo seguir estudiando",
    "voy a dejar todo", "no soporto más", "quiero abandonar todo"
]

textos_motivacion = [
    # Frases CLARAMENTE positivas sobre la escuela
    "me gusta la escuela", "me gusta estudiar mucho", "quiero mejorar mis notas",
    "busco motivación para estudiar", "cómo ser mejor estudiante", "técnicas de estudio efectivas",
    "quiero seguir estudiando siempre", "me encanta aprender", "disfruto ir a clases",
    "me gusta mi carrera", "estoy feliz estudiando", "me gusta venir a la universidad",
    "quiero ser buen estudiante", "me motiva estudiar", "me siento bien en la escuela",
    "amo mi carrera", "me gusta mucho la universidad", "disfruto aprendiendo",
    "me encanta ir a clases", "quiero destacar en mis estudios", "me siento motivado",
    "estoy contento con mis estudios", "me gusta todo de la escuela"
]

textos_positivos_escuela = [
    # Comentarios positivos generales sobre la experiencia escolar
    "la escuela está bien", "me gusta venir aquí", "la universidad es buena",
    "estoy bien en la escuela", "me siento cómodo aquí", "la escuela es interesante",
    "me agrada la universidad", "disfruto estar aquí", "la escuela me parece bien",
    "me gusta el ambiente", "la universidad está padre", "me siento a gusto",
    "la escuela es genial", "me divierte estar aquí", "me gusta el campus"
]

# Combinar todos los textos
todos_los_textos = (textos_matematicas + textos_fisica + textos_quimica + 
                   textos_programacion + textos_desercion + textos_motivacion + 
                   textos_positivos_escuela)

todas_las_etiquetas = (["matematicas"] * len(textos_matematicas) +
                      ["fisica"] * len(textos_fisica) +
                      ["quimica"] * len(textos_quimica) +
                      ["programacion"] * len(textos_programacion) +
                      ["desercion"] * len(textos_desercion) +
                      ["motivacion"] * len(textos_motivacion) +
                      ["positivo"] * len(textos_positivos_escuela))

# Entrenar modelo con mejor configuración
X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas

# Usar configuración más precisa para el modelo
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X, y)

print(f"Modelo entrenado con {len(todos_los_textos)} ejemplos y {len(set(y))} categorías")

# Consejos simples por categoría
consejos = {
    "matematicas": [
        "Te recomiendo practicar ejercicios similares todos los días.",
        "Busca videos explicativos en línea para conceptos difíciles.",
        "Forma grupos de estudio con compañeros.",
        "Consulta con tu profesor durante sus horas de oficina."
    ],
    "fisica": [
        "Conecta los conceptos físicos con ejemplos de la vida real.",
        "Dibuja diagramas para visualizar los problemas.",
        "Revisa las unidades de medida cuidadosamente.",
        "Practica con ejercicios paso a paso."
    ],
    "quimica": [
        "Memoriza la tabla periódica gradualmente.",
        "Practica balancear ecuaciones químicas.",
        "Relaciona la química con ejemplos cotidianos.",
        "Usa modelos moleculares para visualizar."
    ],
    "programacion": [
        "Programa todos los días, aunque sea poco tiempo.",
        "Empieza con proyectos pequeños.",
        "Lee código de otros programadores.",
        "No tengas miedo a los errores, son parte del aprendizaje."
    ],
    "desercion": [
        "Habla con un consejero académico sobre tus sentimientos.",
        "Recuerda tus metas originales y por qué comenzaste.",
        "Busca apoyo en familia, amigos o grupos de estudio.",
        "Considera diferentes métodos de estudio que se adapten a ti."
    ],
    "motivacion": [
        "¡Excelente actitud! Sigue esforzándote.",
        "Establece metas pequeñas y celebra cada logro.",
        "Crea un ambiente de estudio ordenado.",
        "Rodéate de compañeros motivados."
    ],
    "positivo": [
        "¡Qué bueno escuchar comentarios positivos sobre la escuela!",
        "Es genial que tengas una buena experiencia universitaria.",
        "Mantén esa actitud positiva, te ayudará mucho en tus estudios.",
        "Tu perspectiva positiva es muy valiosa para tu éxito académico."
    ]
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    try:
        data = request.json
        texto = data.get("texto", "").lower()
        
        # Verificación adicional para palabras clave positivas/negativas
        palabras_muy_positivas = ["me gusta", "me encanta", "amo", "disfruto", "genial", "bueno", "bien", "contento", "feliz"]
        palabras_muy_negativas = ["odio", "no aguanto", "quiero dejar", "abandonar", "salir", "renunciar", "no soporto"]
        
        # Contar palabras positivas y negativas
        positivas = sum(1 for palabra in palabras_muy_positivas if palabra in texto)
        negativas = sum(1 for palabra in palabras_muy_negativas if palabra in texto)
        
        # Clasificación por ML
        X_test = vectorizer.transform([texto])
        pred = model.predict(X_test)[0]
        probabilidades = model.predict_proba(X_test)[0]
        confianza = max(probabilidades)
        
        # Corrección inteligente: si hay más palabras positivas y predice deserción, corregir
        if pred == "desercion" and positivas > negativas and positivas > 0:
            pred = "positivo"
            confianza = 0.85
        
        # Si es claramente positivo sobre la escuela, asegurar clasificación correcta
        if any(frase in texto for frase in ["me gusta la escuela", "me gusta estudiar", "me gusta la universidad"]):
            pred = "positivo"
            confianza = 0.90
            
        # Elegir un consejo según categoría
        consejo = random.choice(consejos.get(pred, consejos["motivacion"]))
        
        # Respuesta más natural
        respuestas_naturales = {
            "matematicas": "Veo que necesitas ayuda con matemáticas",
            "fisica": "Identifico una consulta sobre física", 
            "quimica": "Detecto que tienes dudas de química",
            "programacion": "Reconozco una pregunta sobre programación",
            "desercion": "Noto que estás pasando por un momento difícil con tus estudios",
            "motivacion": "¡Qué bueno verte tan motivado con tus estudios!",
            "positivo": "¡Me alegra escuchar comentarios tan positivos sobre la escuela!"
        }
        
        respuesta = respuestas_naturales.get(pred, f"He identificado tu consulta sobre: {pred}")
        
        return jsonify({
            "respuesta": respuesta,
            "consejo": consejo,
            "categoria": pred,
            "confianza": f"{confianza:.0%}"
        })
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "respuesta": "He recibido tu consulta académica.",
            "consejo": "Te recomiendo consultar con tu profesor o tutor académico."
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)