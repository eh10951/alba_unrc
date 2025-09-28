from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask_cors import CORS
import os
import random
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuración para servir archivos estáticos
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../frontend', filename)

# Dataset expandido con múltiples categorías académicas
dataset = {
    # MATEMÁTICAS
    "matematicas": [
        "necesito ayuda con álgebra", "no entiendo las ecuaciones", "me cuesta geometría",
        "ayuda con cálculo diferencial", "problemas con trigonometría", "dificultades en estadística",
        "no comprendo las derivadas", "integrales son complicadas", "límites matemáticos",
        "resolver sistemas de ecuaciones", "factorización de polinomios", "matrices y determinantes",
        "números complejos", "probabilidad y combinatoria", "logaritmos y exponenciales"
    ],
    
    # FÍSICA
    "fisica": [
        "no entiendo la mecánica", "problemas con fuerzas", "cinemática es difícil",
        "ayuda con termodinámica", "electromagnetismo complicado", "óptica y ondas",
        "física cuántica", "relatividad especial", "trabajo y energía",
        "momento angular", "fluidos y presión", "oscilaciones armónicas",
        "campo eléctrico", "circuitos eléctricos", "leyes de Newton"
    ],
    
    # QUÍMICA
    "quimica": [
        "tabla periódica", "enlaces químicos", "reacciones químicas",
        "estequiometría", "equilibrio químico", "ácidos y bases",
        "química orgánica", "compuestos químicos", "mol y molaridad",
        "termoquímica", "cinética química", "electroquímica",
        "hibridación", "geometría molecular", "ph y poh"
    ],
    
    # PROGRAMACIÓN
    "programacion": [
        "aprender a programar", "algoritmos difíciles", "estructuras de datos",
        "orientación a objetos", "bases de datos", "desarrollo web",
        "python javascript java", "debugging errores", "lógica de programación",
        "funciones y métodos", "arrays y listas", "recursividad",
        "programación funcional", "patrones de diseño", "apis y servicios"
    ],
    
    # BIOLOGÍA
    "biologia": [
        "célula y organelos", "genética y adn", "evolución especies",
        "ecosistemas", "fotosíntesis", "respiración celular",
        "anatomía humana", "fisiología", "microbiología",
        "botánica plantas", "zoología animales", "biotecnología",
        "herencia genética", "mutaciones", "biodiversidad"
    ],
    
    # HISTORIA
    "historia": [
        "historia mundial", "civilizaciones antiguas", "guerras mundiales",
        "revolución francesa", "edad media", "renacimiento",
        "historia de méxico", "independencia", "revolución mexicana",
        "culturas prehispánicas", "conquista española", "época colonial",
        "historia contemporánea", "siglo xx", "historia moderna"
    ],
    
    # ESPAÑOL/LITERATURA
    "espanol": [
        "gramática española", "literatura mexicana", "análisis literario",
        "ortografía y redacción", "figuras retóricas", "géneros literarios",
        "poesía y narrativa", "ensayo académico", "comprensión lectora",
        "sintaxis y semántica", "conjugación verbal", "escritura creativa",
        "crítica literaria", "movimientos literarios", "autores clásicos"
    ],
    
    # INGLÉS
    "ingles": [
        "english grammar", "conversation practice", "writing skills",
        "reading comprehension", "vocabulary building", "pronunciation",
        "verb tenses", "phrasal verbs", "business english",
        "academic writing", "listening skills", "speaking fluency",
        "translation exercises", "english literature", "idioms and expressions"
    ],
    
    # DESERCIÓN ACADÉMICA
    "desercion": [
        "quiero dejar la escuela", "no tengo motivación", "me siento cansado de estudiar",
        "ya no quiero ir a clases", "abandonar mis estudios", "no me sirve estudiar",
        "odio la universidad", "no veo el punto", "quiero dejar todo",
        "me rindo con los estudios", "demasiado difícil", "no puedo más",
        "quiero trabajar en lugar de estudiar", "la carrera no es para mí", "perdiendo el tiempo"
    ],
    
    # MOTIVACIÓN ACADÉMICA
    "motivacion": [
        "me gusta estudiar", "quiero mejorar mis notas", "busco motivación",
        "cómo ser mejor estudiante", "técnicas de estudio", "organizar mi tiempo",
        "hábitos de estudio", "concentración", "memoria y aprendizaje",
        "superar la procrastinación", "metas académicas", "disciplina estudiantil",
        "autoestima académica", "confianza en mí mismo", "perseverancia"
    ]
}

# Crear dataset de entrenamiento
textos_entrenamiento = []
etiquetas_entrenamiento = []

for categoria, textos in dataset.items():
    for texto in textos:
        textos_entrenamiento.append(texto)
        etiquetas_entrenamiento.append(categoria)

# Usar TF-IDF en lugar de CountVectorizer (mejor para clasificación de texto)
vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    textos_entrenamiento, etiquetas_entrenamiento, 
    test_size=0.2, random_state=42
)

# Entrenar modelo
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Usar Random Forest (mejor que Logistic Regression para múltiples categorías)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

# Calcular precisión
y_pred = model.predict(X_test_vectorized)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2%}")

# Sistema de respuestas profesionales por categoría
respuestas_sistema = {
    "matematicas": {
        "respuesta": "He identificado una consulta académica sobre matemáticas. Te proporciono recursos especializados para tu desarrollo académico.",
        "consejos": [
            "📚 RECURSOS ACADÉMICOS: Consulta la biblioteca digital de tu universidad y accede a libros especializados como 'Álgebra Lineal' de Grossman o 'Cálculo' de Stewart. Solicita asesoría en el Centro de Apoyo Académico.",
            "🎓 TUTORÍA UNIVERSITARIA: Inscríbete en el programa de tutorías peer-to-peer de tu facultad. Los estudiantes de semestres avanzados pueden explicarte conceptos desde su experiencia reciente.",
            "💻 PLATAFORMAS ESPECIALIZADAS: Utiliza WolframAlpha para verificación de cálculos, Khan Academy para fundamentos, y MIT OpenCourseWare para contenido universitario avanzado.",
            "👨‍🏫 HORAS DE OFICINA: Programa citas regulares con tu profesor durante sus horas de oficina. Prepara preguntas específicas y lleva ejercicios resueltos para revisar tu metodología.",
            "📊 EVALUACIÓN DIAGNÓSTICA: Solicita en servicios estudiantiles una evaluación de tu nivel matemático para identificar áreas específicas de refuerzo y crear un plan de estudio personalizado."
        ]
    },
    "fisica": {
        "respuesta": "Reconozco una consulta de física aplicada. Te dirijo hacia recursos académicos especializados en ciencias físicas.",
        "consejos": [
            "🔬 LABORATORIOS UNIVERSITARIOS: Solicita acceso a laboratorios de física fuera del horario de clase. La práctica experimental consolida la teoría. Contacta al coordinador de laboratorios para sesiones adicionales.",
            "📖 BIBLIOGRAFÍA ESPECIALIZADA: Consulta 'Physics for Scientists and Engineers' de Serway o 'University Physics' de Young & Freedman en la biblioteca universitaria. Estos textos tienen ejemplos paso a paso.",
            "🎯 GRUPOS DE INVESTIGACIÓN: Acércate a grupos de investigación en física de tu universidad. Pueden ofrecerte proyectos de iniciación científica que refuercen tu aprendizaje teórico.",
            "💡 SIMULADORES PROFESIONALES: Utiliza PhET Interactive Simulations (Universidad de Colorado) y Algodoo para visualizar conceptos. Estos recursos son utilizados por universidades internacionales.",
            "🏆 COMPETENCIAS ACADÉMICAS: Participa en olimpiadas de física universitarias. La preparación intensiva mejora significativamente tu comprensión de principios fundamentales."
        ]
    },
    "quimica": {
        "respuesta": "Detecto una consulta en ciencias químicas. Te proporciono recursos académicos profesionales para química universitaria.",
        "consejos": [
            "🧪 RECURSOS ESPECIALIZADOS: Accede a ChemSpider y SciFinder Scholar através de la biblioteca universitaria. Consulta 'Química General' de Petrucci y 'Química Orgánica' de Morrison & Boyd.",
            "⚗️ LABORATORIO AVANZADO: Solicita al coordinador de laboratorios químicos acceso a equipos como espectrómetros IR y RMN. La práctica instrumental es crucial para tu formación profesional.",
            "🔬 INVESTIGACIÓN APLICADA: Contacta profesores con líneas de investigación en química para oportunidades de servicio social o tesis. La investigación consolida conocimientos teóricos.",
            "💻 SOFTWARE PROFESIONAL: Aprende ChemDraw, Gaussian y Spartan para modelado molecular. Estos programas son estándar en la industria química y farmacéutica.",
            "🏭 VINCULACIÓN INDUSTRIAL: Participa en visitas a industrias químicas organizadas por tu facultad. Observar procesos reales ayuda a entender aplicaciones comerciales de la química."
        ]
    },
    "programacion": {
        "respuesta": "Identifico una consulta sobre desarrollo de software. Te proporciono recursos profesionales para tu carrera en tecnología.",
        "consejos": [
            "💼 DESARROLLO PROFESIONAL: Crea tu perfil en LinkedIn con proyectos de GitHub. Conecta con el Centro de Vinculación Laboral de tu universidad para oportunidades de prácticas profesionales en empresas tech.",
            "🏢 CERTIFICACIONES INDUSTRIA: Obtén certificaciones gratuitas: Google IT Support, Microsoft Azure Fundamentals, AWS Cloud Practitioner. Estas credenciales son valoradas por empleadores.",
            "👥 NETWORKING ACADÉMICO: Únete al capítulo estudiantil de ACM o IEEE en tu universidad. Participa en hackathons y coding competitions para expandir tu red profesional.",
            "🚀 PROYECTOS COLABORATIVOS: Contribuye a proyectos open source relacionados con tu área de estudio. Esto demuestra habilidades profesionales a futuros empleadores.",
            "📊 METODOLOGÍAS ÁGILES: Aprende Scrum y metodologías ágiles en el Centro de Emprendimiento universitario. Estas habilidades son esenciales en equipos de desarrollo profesional."
        ]
    },
    "biologia": {
        "respuesta": "Detecto consulta en ciencias biológicas. Te dirijo hacia recursos académicos especializados en investigación biológica.",
        "consejos": [
            "🔬 INVESTIGACIÓN CIENTÍFICA: Contacta laboratorios de investigación biológica de tu universidad para oportunidades de servicio social o proyectos de titulación en áreas como biotecnología, ecología o biomedicina.",
            "📚 BASES DE DATOS CIENTÍFICAS: Accede a PubMed, Web of Science y Scopus través de la biblioteca universitaria para consultar artículos científicos actualizados y metodologías de investigación.",
            "🧬 COLABORACIÓN INTERDISCIPLINARIA: Participa en proyectos conjuntos con facultades de medicina, ingeniería biomédica y ciencias ambientales para ampliar tu perspectiva profesional.",
            "🌿 TRABAJO DE CAMPO: Inscríbete en estaciones biológicas y reservas naturales afiliadas a tu universidad para experiencia práctica en taxonomía, ecología y conservación.",
            "🏥 VINCULACIÓN CLÍNICA: Explora convenios con hospitales universitarios e institutos de investigación médica para aplicaciones clínicas de conocimientos biológicos."
        ]
    },
    "historia": {
        "respuesta": "Reconozco consulta en ciencias históricas. Te proporciono recursos académicos para investigación historiográfica profesional.",
        "consejos": [
            "📖 FUENTES PRIMARIAS: Utiliza el Archivo Histórico universitario y repositorios digitales como JSTOR y Project MUSE para acceder a documentos originales y fuentes primarias.",
            "🏛️ COLABORACIÓN INSTITUCIONAL: Contacta museos, archivos nacionales y centros de investigación histórica para proyectos de investigación y prácticas profesionales.",
            "📝 METODOLOGÍA HISTORIOGRÁFICA: Inscríbete en seminarios de metodología de la investigación histórica y paleografía ofrecidos por el posgrado en historia.",
            "🌍 PERSPECTIVA GLOBAL: Participa en congresos estudiantiles de historia y programas de intercambio académico para ampliar tu visión historiográfica internacional.",
            "💻 HUMANIDADES DIGITALES: Aprende herramientas digitales como Omeka, Zotero y GIS para análisis espacial e investigación histórica contemporánea."
        ]
    },
    "espanol": {
        "respuesta": "Identifico consulta en estudios lingüísticos y literarios. Te dirijo hacia recursos académicos especializados en filología hispánica.",
        "consejos": [
            "📚 CORPUS LINGÜÍSTICOS: Utiliza CORDE, CREA y Corpus del Español a través de la biblioteca universitaria para análisis lingüístico y evolución del idioma español.",
            "✍️ TALLERES ESPECIALIZADOS: Participa en talleres de escritura académica, corrección de estilo y edición de textos ofrecidos por el Centro de Escritura universitario.",
            "🎭 GRUPOS LITERARIOS: Únete a grupos de creación literaria, revistas estudiantiles y círculos de lectura coordinados por la facultad de letras.",
            "🏆 CONCURSOS ACADÉMICOS: Participa en concursos de ensayo, cuento y poesía universitarios. Estos reconocimientos fortalecen tu currículum académico y profesional.",
            "📖 INVESTIGACIÓN LITERARIA: Colabora con profesores en proyectos de investigación sobre literatura hispanoamericana, análisis del discurso y crítica literaria."
        ]
    },
    "ingles": {
        "respuesta": "I've identified an English language academic inquiry. I'm providing you with professional resources for advanced English proficiency.",
        "consejos": [
            "🎓 CERTIFICACIONES INTERNACIONALES: Prepárate para exámenes TOEFL, IELTS o Cambridge English a través del Centro de Idiomas universitario. Estas certificaciones son requisito para posgrados internacionales.",
            "🌍 PROGRAMAS DE INTERCAMBIO: Solicita información sobre convenios internacionales, programas de movilidad estudiantil y becas para estudios en países anglófonos.",
            "📝 ESCRITURA ACADÉMICA: Inscríbete en cursos de Academic Writing y English for Specific Purposes ofrecidos por el departamento de lenguas extranjeras.",
            "💼 ENGLISH FOR PROFESSIONAL PURPOSES: Desarrolla competencias en Business English, Technical Writing y Presentation Skills valoradas en el mercado laboral internacional.",
            "👥 CONVERSATORIOS INTERNACIONALES: Participa en Model UN, debates académicos y conferencias internacionales organizadas por relaciones internacionales universitarias."
        ]
    },
    "desercion": {
        "respuesta": "He detectado indicadores de riesgo académico. Te proporciono recursos institucionales especializados para apoyo estudiantil.",
        "consejos": [
            "🏥 BIENESTAR UNIVERSITARIO: Agenda cita inmediata con el Departamento de Bienestar Estudiantil. Ofrecen apoyo psicológico profesional, orientación vocacional y programas de retención estudiantil sin costo.",
            "📋 ASESORÍA ACADÉMICA FORMAL: Solicita evaluación con tu Coordinador Académico para analizar tu carga académica, historial de calificaciones y opciones de reubicación curricular.",
            "💰 APOYO FINANCIERO: Consulta en Servicios Escolares sobre becas de apoyo económico, programas trabajo-estudio y opciones de financiamiento que reduzcan presión económica.",
            "🎯 REORIENTACIÓN VOCACIONAL: El Centro de Orientación Vocacional puede aplicarte pruebas psicométricas para confirmar si tu carrera actual se alinea con tus aptitudes e intereses.",
            "👥 GRUPOS DE APOYO: Participa en grupos de apoyo entre pares coordinados por Trabajo Social universitario. Conectar con estudiantes en situaciones similares reduce el aislamiento académico."
        ]
    },
    "motivacion": {
        "respuesta": "Reconozco una consulta sobre optimización del rendimiento académico. Te proporciono estrategias basadas en evidencia científica.",
        "consejos": [
            "🧠 TÉCNICAS METACOGNITIVAS: Implementa técnicas de aprendizaje activo validadas: técnica Feynman, mapas mentales y espaciado distribuido. El Centro de Apoyo al Aprendizaje ofrece talleres especializados.",
            "📈 PLAN DE DESARROLLO ACADÉMICO: Trabaja con tu tutor académico para crear un Plan Individual de Formación (PIF) con objetivos SMART y métricas de seguimiento semestral.",
            "🏆 RECONOCIMIENTO INSTITUCIONAL: Postúlate a programas de excelencia académica, becas al mérito y reconocimientos estudiantiles. Estos logros fortalecen autoeficacia y motivación intrínseca.",
            "🌟 LIDERAZGO ESTUDIANTIL: Participa en gobierno estudiantil, sociedades académicas y proyectos de extensión universitaria. El liderazgo desarrolla competencias transversales valoradas profesionalmente.",
            "📊 SEGUIMIENTO PSICOEDUCATIVO: Utiliza herramientas de autoevaluación del Centro de Desarrollo Estudiantil para monitorear tu progreso académico y ajustar estrategias de estudio basándote en evidencia."
        ]
    }
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.json
    texto = data.get("texto", "")
    
    # Vectorizar el texto de entrada
    X_test = vectorizer.transform([texto])
    
    # Predecir la categoría
    pred = model.predict(X_test)[0]
    
    # Obtener probabilidades para mostrar confianza
    probabilidades = model.predict_proba(X_test)[0]
    confianza = max(probabilidades)
    
    # Obtener respuesta y consejo personalizado
    if pred in respuestas_sistema:
        respuesta = respuestas_sistema[pred]["respuesta"]
        consejo = random.choice(respuestas_sistema[pred]["consejos"])
    else:
        # Fallback para categorías no definidas
        respuesta = f"He identificado tu consulta como: {pred}"
        consejo = "Te recomiendo consultar con tu profesor o buscar recursos adicionales sobre este tema."
    
    return jsonify({
        "respuesta": respuesta,
        "consejo": consejo,
        "categoria": pred,
        "confianza": f"{confianza:.1%}"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
