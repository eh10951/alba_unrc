from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Entrenamiento mejorado con más datos
vectorizer = CountVectorizer()

# Ejemplos de deserción
textos_desercion = [
    "quiero dejar la escuela", 
    "no tengo motivación para estudiar", 
    "me siento cansado de estudiar",
    "ya no quiero ir a clases",
    "pienso abandonar mis estudios",
    "no me sirve estudiar",
    "odio ir a la universidad",
    "no veo el punto de seguir estudiando",
    "quiero dejar todo y trabajar",
    "me rindo con los estudios"
]

# Ejemplos de NO deserción (incluyendo comentarios positivos sobre materias)
textos_no_desercion = [
    "me gusta estudiar", 
    "me esfuerzo por mejorar", 
    "quiero terminar mi carrera",
    "me gustan las materias que estoy viendo",
    "disfruto aprendiendo nuevas cosas",
    "me motiva sacar buenas notas",
    "quiero ser un buen profesional",
    "me gusta mi carrera",
    "estoy aprendiendo mucho",
    "me siento orgulloso de mis estudios",
    "me gustan las clases",
    "quiero seguir estudiando",
    "me gusta venir a la universidad",
    "disfruto las materias",
    "me parece interesante lo que estudio"
]

# Combinar todos los textos
todos_los_textos = textos_desercion + textos_no_desercion
todas_las_etiquetas = ["desercion"] * len(textos_desercion) + ["no_desercion"] * len(textos_no_desercion)

X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas

model = LogisticRegression()
model.fit(X, y)

# Consejos asociados a cada categoría
consejos = {
    "desercion": [
        "Recuerda que pedir ayuda a un maestro o consejero puede darte nuevas ideas.",
        "Piensa en tus metas a futuro, la educación es un paso importante.",
        "No estás solo, busca apoyo en tus compañeros o familia."
    ],
    "no_desercion": [
        "¡Excelente actitud! Sigue esforzándote, vas por buen camino.",
        "Mantén esa motivación, cada día es una oportunidad de aprender.",
        "Tu esfuerzo te llevará lejos, sigue así."
    ]
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.json
    texto = data.get("texto", "")
    X_test = vectorizer.transform([texto])
    pred = model.predict(X_test)[0]

    # Elegir un consejo según categoría
    import random
    consejo = random.choice(consejos[pred])

    return jsonify({
        "respuesta": f"Clasificación: {pred}",
        "consejo": consejo
    })

if __name__ == "__main__":
    app.run(debug=True)
