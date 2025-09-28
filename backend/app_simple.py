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
    "quiero dejar la escuela", "no tengo motivación", "me siento cansado de estudiar",
    "ya no quiero ir a clases", "abandonar mis estudios", "no me sirve estudiar"
]

textos_motivacion = [
    "me gusta estudiar", "quiero mejorar mis notas", "busco motivación",
    "cómo ser mejor estudiante", "técnicas de estudio", "quiero seguir estudiando"
]

# Combinar todos los textos
todos_los_textos = (textos_matematicas + textos_fisica + textos_quimica + 
                   textos_programacion + textos_desercion + textos_motivacion)

todas_las_etiquetas = (["matematicas"] * len(textos_matematicas) +
                      ["fisica"] * len(textos_fisica) +
                      ["quimica"] * len(textos_quimica) +
                      ["programacion"] * len(textos_programacion) +
                      ["desercion"] * len(textos_desercion) +
                      ["motivacion"] * len(textos_motivacion))

# Entrenar modelo
X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas

model = LogisticRegression()
model.fit(X, y)

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
    ]
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    try:
        data = request.json
        texto = data.get("texto", "")
        
        X_test = vectorizer.transform([texto])
        pred = model.predict(X_test)[0]
        
        # Elegir un consejo según categoría
        consejo = random.choice(consejos.get(pred, consejos["motivacion"]))
        
        return jsonify({
            "respuesta": f"He identificado tu consulta sobre: {pred}",
            "consejo": consejo
        })
    
    except Exception as e:
        return jsonify({
            "respuesta": "He recibido tu consulta académica.",
            "consejo": "Te recomiendo consultar con tu profesor o tutor académico."
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)