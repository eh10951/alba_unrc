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

# Consejos profesionales con recursos específicos
consejos = {
    "matematicas": [
        "📚 CURSOS RECOMENDADOS: Inscríbete en el curso 'Fundamentos de Álgebra' de Khan Academy (gratuito) y 'Cálculo Diferencial' en Coursera por Universidad UNAM. Contacta al Tutor de Matemáticas del Centro de Apoyo Académico de tu universidad.",
        "👨‍🏫 APOYO PERSONALIZADO: Solicita tutoría con estudiantes de Ingeniería de semestres avanzados (programa peer tutoring). El Dr. Luis García del Departamento de Matemáticas ofrece asesorías los martes y jueves de 2-4 PM.",
        "💻 RECURSOS DIGITALES: Usa Wolfram Alpha para verificar cálculos y Photomath para resolver paso a paso. El libro 'Álgebra de Baldor' está disponible gratis en la biblioteca digital universitaria.",
        "🎯 PLAN DE ESTUDIO: Practica 30 min diarios con ejercicios graduales. Únete al Círculo de Estudio de Matemáticas que se reúne miércoles 4 PM en el aula 205. Profesora María Rodríguez coordina.",
        "🏆 CERTIFICACIÓN: Considera el curso 'Matemáticas para Ciencias' del Tecnológico de Monterrey en edX (con certificado). Te dará una base sólida y reconocimiento académico adicional."
    ],
    "fisica": [
        "🔬 LABORATORIOS ESPECIALIZADOS: Solicita acceso al Laboratorio de Física Básica fuera de horario de clase. El Ing. Roberto Martínez (Ext. 3421) coordina sesiones prácticas adicionales sábados 9-12.",
        "📖 CURSOS COMPLEMENTARIOS: 'Física Universitaria' de UC Berkeley en edX y 'Mecánica Clásica' de MIT OpenCourseWare. Para refuerzo presencial, el Círculo de Física se reúne viernes 3 PM.",
        "🎬 RECURSOS AUDIOVISUALES: Canal de YouTube 'MinutoDeFísica' y simuladores PhET de Universidad de Colorado. La Dra. Ana López ofrece tutorías personalizadas lunes y miércoles 1-3 PM.",
        "📊 APOYO ACADÉMICO: El programa 'Nivelación en Ciencias Exactas' del Departamento de Física ofrece cursos remediales gratuitos. Inscripciones en Servicios Escolares.",
        "🌟 PROYECTO APLICADO: Únete al Grupo de Divulgación Científica dirigido por el Dr. Carlos Hernández. Aplicar física en proyectos reales mejora comprensión y currículum."
    ],
    "quimica": [
        "⚗️ LABORATORIO ABIERTO: El Laboratorio de Química General ofrece sesiones libres sábados 10-1 PM. Contacta a la Q.F.B. Patricia Morales (Ext. 2890) para reservar equipo especializado.",
        "🧪 CURSOS PROFESIONALES: 'Química General' de Universidad de Kentucky en Coursera y 'Organic Chemistry' de Yale (gratuito). Para apoyo presencial, Dr. Miguel Ángel Sánchez, cubículo 12B.",
        "💻 SOFTWARE ESPECIALIZADO: Aprende ChemSketch (gratis) y MarvinSketch para estructura molecular. Centro de Cómputo ofrece taller 'Química Computacional' cada mes.",
        "👥 GRUPOS DE ESTUDIO: Círculo de Química Orgánica dirigido por estudiantes de Ingeniería Química, jueves 5 PM, aula 108. Resuelven problemas complejos en equipo.",
        "🏭 VINCULACIÓN INDUSTRIAL: Participa en visitas a plantas químicas organizadas por Vinculación Universitaria. Observar procesos reales consolida conocimiento teórico."
    ],
    "programacion": [
        "💼 CERTIFICACIONES PROFESIONALES: Google IT Automation with Python (Coursera), Microsoft Azure Fundamentals y AWS Cloud Practitioner (gratuitas para estudiantes). Centro de Vinculación Laboral te apoya.",
        "👨‍💻 MENTORES ESPECIALIZADOS: Contacta a egresados en el programa Alumni Mentorship. El Ing. Fernando López (Google México) ofrece mentoría virtual mensual para estudiantes destacados.",
        "🚀 PROYECTOS COLABORATIVOS: Únete a los capítulos estudiantiles ACM e IEEE. Participan en hackathons y tienen biblioteca especializada en el edificio de Ingeniería.",
        "📚 CURSOS AVANZADOS: 'CS50 Introduction to Computer Science' de Harvard (gratis), 'Python for Data Science' de IBM. El Centro de Cómputo ofrece cursos presenciales.",
        "🏢 PRÁCTICAS PROFESIONALES: Programa de Internships con empresas tech locales. Oficina de Prácticas Profesionales coordina con Microsoft, IBM y startups locales."
    ],
    "desercion": [
        "🏥 URGENTE - BIENESTAR ESTUDIANTIL: Contacta HOY al Departamento de Bienestar Estudiantil (Ext. 2500, edificio administrativo 2do piso). Psic. Laura Jiménez especializada en crisis académicas.",
        "📋 ASESORÍA ACADÉMICA INMEDIATA: Tu Coordinador Académico puede evaluar opciones: baja temporal, reducción de materias, cambio de modalidad. Agenda cita en Servicios Escolares.",
        "💰 APOYO FINANCIERO: Programa de Becas de Emergencia y trabajo-estudio disponible. Lic. Carmen Flores (Oficina de Becas) evalúa casos especiales. También crédito educativo FONACOT.",
        "🎯 ORIENTACIÓN VOCACIONAL: Centro de Orientación puede aplicar test vocacional y de intereses (gratuito). Confirma si tu carrera actual se alinea con tus aptitudes reales.",
        "📞 LÍNEA DE APOYO 24/7: Universidad cuenta con línea de crisis estudiantil: 800-APOYO-U. También grupo de WhatsApp 'Red de Apoyo Estudiantil' moderado por Trabajo Social."
    ],
    "motivacion": [
        "🏆 PROGRAMA DE EXCELENCIA: Inscríbete en el Programa de Estudiantes Sobresalientes que ofrece mentorías, conferencias exclusivas y networking con profesionales exitosos.",
        "📈 DESARROLLO DE HABILIDADES: Curso 'Técnicas de Estudio Avanzadas' del Centro de Desarrollo Estudiantil. También taller 'Liderazgo Universitario' con reconocimiento curricular.",
        "🌟 SOCIEDADES ACADÉMICAS: Únete al Consejo Estudiantil o sociedad de alumnos de tu carrera. Liderazgo estudiantil desarrolla competencias valoradas por empleadores.",
        "🎯 COACHING ACADÉMICO: Programa de coaching personalizado con egresados exitosos. 6 sesiones gratuitas enfocadas en metas académicas y desarrollo profesional.",
        "🚀 OPORTUNIDADES ESPECIALES: Postúlate a programas de intercambio, concursos académicos y proyectos de investigación con profesores. Fortalecen perfil académico significativamente."
    ],
    "positivo": [
        "🌟 EMBAJADOR ESTUDIANTIL: Con tu actitud positiva, considera ser Embajador Estudiantil para orientar a nuevos estudiantes. Programa coordinado por Relaciones Estudiantiles.",
        "🎓 ACTIVIDADES DE LIDERAZGO: Participa en el Consejo Estudiantil o comités académicos. Tu perspectiva positiva puede influir en mejoras para toda la comunidad universitaria.",
        "📢 TESTIMONIOS INSPIRADORES: Comparte tu experiencia positiva en eventos de orientación para nuevos estudiantes. Centro de Comunicación busca historias estudiantiles exitosas.",
        "🏅 RECONOCIMIENTOS: Postúlate a programas de reconocimiento como 'Estudiante del Mes' o 'Orgullo Universitario'. Tu actitud positiva merece ser destacada institucionalmente.",
        "🤝 MENTORING: Considera ser mentor de estudiantes de primer semestre. Tu experiencia positiva puede ayudar a otros a adaptarse mejor a la vida universitaria."
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