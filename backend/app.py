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
    "voy a dejar todo", "no soporto más", "quiero abandonar todo",
    "no vale la pena estudiar", "esto es una pérdida de tiempo", "odio esta carrera",
    "me quiero cambiar de carrera", "ya no aguanto más clases", "estoy harto de estudiar",
    "no sirvo para esto", "mejor me salgo", "no tiene sentido continuar",
    "estoy perdiendo el tiempo aquí", "no me gusta nada de esto", "todo me sale mal",
    "no entiendo nada y ya me cansé", "prefiero trabajar que estudiar", "esto es muy difícil para mí"
]

textos_motivacion = [
    # Frases CLARAMENTE positivas sobre la escuela y estudio
    "busco motivación para estudiar", "cómo ser mejor estudiante", "técnicas de estudio efectivas",
    "quiero mejorar mis hábitos de estudio", "necesito consejos para estudiar mejor",
    "cómo organizarme mejor", "quiero ser más disciplinado", "necesito técnicas de concentración",
    "cómo manejar mi tiempo de estudio", "quiero ser más productivo estudiando",
    "consejos para no procrastinar", "cómo mantenerme motivado", "estrategias de aprendizaje",
    "cómo mejorar mi rendimiento académico", "técnicas de memorización",
    "cómo preparar mejor los exámenes", "consejos para tomar mejores apuntes",
    "cómo superar la pereza para estudiar", "métodos de estudio efectivos"
]

textos_positivos_escuela = [
    # Comentarios EXPLÍCITAMENTE positivos sobre la experiencia escolar
    "me gusta la escuela", "me gusta estudiar mucho", "me gusta la universidad",
    "me encanta aprender", "disfruto ir a clases", "me gusta mi carrera",
    "estoy feliz estudiando", "me gusta venir a la universidad", "amo mi carrera",
    "me gusta mucho la universidad", "disfruto aprendiendo", "me encanta ir a clases",
    "quiero destacar en mis estudios", "me siento motivado estudiando",
    "estoy contento con mis estudios", "me gusta todo de la escuela",
    "la escuela está genial", "me gusta venir aquí", "la universidad es buena",
    "estoy bien en la escuela", "me siento cómodo aquí", "la escuela es interesante",
    "me agrada la universidad", "disfruto estar aquí", "me gusta el ambiente",
    "la universidad está padre", "me siento a gusto", "la escuela es genial",
    "me divierte estar aquí", "me gusta el campus", "me gusta estudiar aquí",
    "que buena está la universidad", "me fascina mi carrera", "amo estudiar",
    "me encanta esta universidad", "disfruto mucho las clases", "me gusta aprender",
    "estoy muy contento aquí", "me parece excelente la escuela"
]

# Textos negativos generales (quejas sin intención de abandono)
textos_negativos = [
    "esta materia es muy difícil", "no me gusta esta clase", "el profesor explica mal",
    "esto está muy complicado", "no entiendo nada", "esto es muy aburrido",
    "esta clase es un fastidio", "qué difícil está todo", "no me sale nada bien",
    "estoy muy estresado con los estudios", "tengo muchas tareas", "esto me frustra",
    "no logro concentrarme", "me cuesta mucho trabajo", "esto me desespera",
    "qué complicado está todo", "me siento abrumado", "esto me está costando",
    "no me está yendo bien", "estoy batallando mucho", "esto me tiene estresado",
    # Casos específicos de bajo rendimiento académico
    "estoy reprobando materias", "voy mal en mis calificaciones", "saqué malas notas",
    "reprobé el examen", "me fue mal en el examen", "tengo calificaciones bajas",
    "estoy reprobando", "reprobé la materia", "tengo materias reprobadas",
    "mis notas están muy bajas", "estoy fallando en los estudios", "no paso las materias",
    "tengo puras calificaciones bajas", "me está yendo muy mal", "estoy fracasando",
    "no logro aprobar", "siempre repruebo", "mis calificaciones son terribles",
    "estoy perdiendo materias", "voy reprobando todo", "no puedo aprobar nada"
]

# Combinar todos los textos
todos_los_textos = (textos_matematicas + textos_programacion + textos_desercion + textos_motivacion + 
                   textos_positivos_escuela + textos_negativos)

todas_las_etiquetas = (["matematicas"] * len(textos_matematicas) +
                      ["programacion"] * len(textos_programacion) +
                      ["desercion"] * len(textos_desercion) +
                      ["motivacion"] * len(textos_motivacion) +
                      ["positivo"] * len(textos_positivos_escuela) +
                      ["negativo"] * len(textos_negativos))

# Entrenar modelo con mejor configuración
X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas

# Usar configuración más precisa para el modelo
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X, y)

print(f"Modelo entrenado con {len(todos_los_textos)} ejemplos y {len(set(y))} categorías")

# Consejos profesionales con recursos específicos
consejos = {
    "matematicas": [
        "Cursos recomendados: 'Matemáticas para Ingenierías' y 'Cálculo y Álgebra' en Udemy; muchos son con ejercicios prácticos y certificados.",
        "YouTube: Khan Academy (Español) y JulioProfe para explicaciones paso a paso y ejercicios resueltos.",
        "Pide ayuda: Asiste a horarios de profesor, solicita sesiones de tutoría en el centro de apoyo académico y forma grupos de estudio con compañeros.",
        "Herramientas prácticas: usa GeoGebra, WolframAlpha o Symbolab para comprobar procesos y practicar paso a paso.",
        "Rutina de práctica: resuelve problemas todos los días 30–60 min, empieza por ejercicios básicos y aumenta dificultad progresivamente."
    ],

    "programacion": [
        "Cursos recomendados: 'Complete Python Bootcamp' (Jose Portilla, Udemy) o cursos de algoritmos y estructuras en Udemy/Platzi.",
        "YouTube: freeCodeCamp.org, Traversy Media y canales en español con proyectos prácticos para ver código real.",
        "Pide ayuda: pregunta en horas de oficina del docente, solicita revisiones de código y participa en talleres de laboratorio.",
        "Practica con proyectos: sube ejercicios a GitHub, haz pequeños proyectos (to-do, API, web scraper) y busca feedback en comunidades.",
        "Recursos de depuración: aprende a usar un debugger, lee errores con calma y usa Stack Overflow y foros estudiantiles para dudas concretas."
    ],

    "desercion": [
        "Comunícate YA con Bienestar Estudiantil o el orientador académico de tu facultad para evaluar opciones (reducción de carga, baja temporal, apoyo psicológico).",
        "Habla con tus profesores y coordinador de carrera: pueden ofrecer adaptaciones, plazos extra o alternativas de evaluación.",
        "Busca apoyo financiero: oficina de becas, programas de ayuda de la universidad o trabajo académico temporal para aliviar presión económica.",
        "Apoyo emocional: agenda consulta con el servicio de psicología universitaria; muchos ofrecen atención y estrategias para crisis.",
        "Plan concreto: arma un plan de 4–8 semanas con metas pequeñas (menos materias, tutorías, descanso) y revisa progresos con un tutor o mentor."
    ],

    "motivacion": [
        "Técnica práctica: usa Pomodoro (25/5) y objetivos pequeños diarios; mide progreso semanal para mantener motivación.",
        "Cursos y vídeos cortos: busca playlists de técnicas de estudio en YouTube y cursos de productividad en Udemy.",
        "Pide apoyo: establece reuniones cortas con un tutor o compañero para rendición de cuentas y revisión de metas.",
        "Organización: usa Notion/Trello para planificar tareas y exámenes; prioriza lo urgente/importante y divide tareas grandes.",
        "Refuerzo positivo: celebra pequeñas victorias, registra avances y ajusta objetivos si algo no funciona."
    ],

    "positivo": [
        "Comparte tu experiencia: ofrece mentoría a nuevos alumnos o participa en actividades de orientación estudiantil.",
        "Sigue aprendiendo: aprovecha cursos avanzados en Udemy/YouTube para profundizar y convertir tu entusiasmo en habilidades.",
        "Liderazgo: participa en grupos estudiantiles o proyectos; son buena forma de potenciar CV y redes académicas.",
        "Da feedback constructivo a la facultad sobre lo que funciona para ti para mejorar el entorno de estudio.",
        "Sé referente: ofrece sesiones de repaso informales para compañeros y documenta recursos útiles en un repositorio compartido."
    ],

    "negativo": [
        "Salud mental y manejo de estrés: agenda atención con psicología universitaria y aprende técnicas básicas de respiración/mindfulness.",
        "Habla con tus profesores: pide retroalimentación concreta sobre cómo mejorar y solicita alternativas (entregas, apoyo, ejercicios extras).",
        "Organiza tiempo y prioridades: reduce carga si es necesario, aplica un calendario semanal y evita intentar todo al mismo tiempo.",
        "Tutorización: solicita tutorías o clases particulares para los temas más débiles; muchos campus ofrecen apoyo gratuito.",
        "Pequeños pasos: identifica 1–2 acciones inmediatas (pedir cita con tutor, completar 1 ejercicio difícil, dormir mejor) y repítelas hasta mejorar."
    ]
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    try:
        data = request.json
        texto = data.get("texto", "").lower()
        
        # Sistema de detección mejorado con palabras clave
        palabras_muy_positivas = [
            "me gusta", "me encanta", "amo", "disfruto", "genial", "bueno", "bien", 
            "contento", "feliz", "excelente", "fantástico", "maravilloso", "perfecto",
            "increíble", "fascinante", "divertido", "interesante", "motivado"
        ]
        
        palabras_desercion = [
            "quiero dejar", "voy a abandonar", "me quiero salir", "quiero renunciar",
            "no aguanto más", "no soporto", "odio", "abandonar", "dejar todo",
            "salirme", "cambiarme de carrera", "esto no es para mí"
        ]
        
        palabras_negativas_generales = [
            "difícil", "complicado", "no entiendo", "frustra", "estresado", "abrumado",
            "batallando", "no me sale", "me cuesta", "desespera", "fastidio", "aburrido",
            "reprobando", "reprobé", "repruebo", "calificaciones bajas", "malas notas",
            "fracasando", "fallando", "perdiendo materias", "mal en el examen", "muy mal"
        ]
        
        # Frases completas que indican positividad clara
        frases_muy_positivas = [
            "me gusta la escuela", "me gusta estudiar", "me gusta la universidad",
            "me encanta aprender", "disfruto las clases", "amo mi carrera",
            "me gusta mi carrera", "estoy feliz estudiando", "me motiva estudiar"
        ]
        
        # Frases que SIEMPRE deben ser clasificadas como negativas
        frases_muy_negativas = [
            "estoy reprobando", "reprobé", "tengo malas notas", "saqué malas calificaciones",
            "me fue mal", "estoy fracasando", "no puedo aprobar", "perdí la materia",
            "tengo calificaciones bajas", "voy mal en", "me está yendo mal",
            "no logro aprobar", "siempre repruebo", "no paso las materias"
        ]
        
        # Clasificación por ML primero
        X_test = vectorizer.transform([texto])
        pred = model.predict(X_test)[0]
        probabilidades = model.predict_proba(X_test)[0]
        confianza = max(probabilidades)
        
        # Sistema de corrección inteligente con PRIORIDADES CLARAS
        
        # PRIORIDAD 1: Frases explícitamente NEGATIVAS (máxima prioridad)
        if any(frase in texto for frase in frases_muy_negativas):
            pred = "negativo"
            confianza = 0.98
            print(f"DETECCIÓN NEGATIVA FORZADA: {texto}")
        
        # PRIORIDAD 2: Verificar deserción
        elif any(palabra in texto for palabra in palabras_desercion):
            pred = "desercion"
            confianza = 0.95
            print(f"DETECCIÓN DESERCIÓN: {texto}")
        
        # PRIORIDAD 3: Verificar frases explícitamente positivas
        elif any(frase in texto for frase in frases_muy_positivas):
            pred = "positivo"
            confianza = 0.93
            print(f"DETECCIÓN POSITIVA: {texto}")
            
        # PRIORIDAD 4: Sistema de conteo y análisis
        else:
            # Contar indicadores en el texto
            positivas = sum(1 for palabra in palabras_muy_positivas if palabra in texto)
            desercion_palabras = sum(1 for palabra in palabras_desercion if palabra in texto)
            negativas_generales = sum(1 for palabra in palabras_negativas_generales if palabra in texto)
            
            print(f"CONTEO - Positivas: {positivas}, Negativas: {negativas_generales}, Deserción: {desercion_palabras}")
            
            # Si hay más indicadores negativos que positivos
            if negativas_generales > positivas and negativas_generales > 0:
                if pred not in ["matematicas", "fisica", "quimica", "programacion"]:
                    pred = "negativo"
                    confianza = 0.87
                    print(f"RECLASIFICACIÓN A NEGATIVO por conteo: {texto}")
            
            # Si hay palabras positivas claras sobre la escuela
            elif positivas > 0 and ("escuela" in texto or "universidad" in texto or "estudiar" in texto):
                if not any(neg in texto for neg in frases_muy_negativas):
                    pred = "positivo"
                    confianza = 0.85
                    print(f"RECLASIFICACIÓN A POSITIVO: {texto}")
            
        # Elegir un consejo según categoría
        consejo = random.choice(consejos.get(pred, consejos["motivacion"]))
        
        # Respuesta más natural
        respuestas_naturales = {
            "matematicas": "Puedo aconsejar materiales de estudio y ofrecer consejos sobre ejercicios y recursos para practicar matemáticas.",
            "programacion": "Puedo aconsejar materiales de estudio y sugerir recursos y ejemplos para practicar programación y depuración.",
            "desercion": "Puedo aconsejar recursos y pasos a considerar (orientación estudiantil, apoyo emocional, opciones administrativas) y ofrecer consejos prácticos.",
            "motivacion": "Puedo aconsejar técnicas y materiales para mejorar la motivación (técnicas de estudio, rutinas, recursos de seguimiento).",
            "positivo": "Puedo aconsejar actividades y materiales para aprovechar el interés por el estudio (proyectos, cursos, lecturas recomendadas).",
            "negativo": "Puedo aconsejar recursos y acciones concretas (tutorías, organización, gestión del estrés) para mejorar la situación académica."
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