from flask import Flask, request, jsonify, send_from_directory  # Framework web y utilidades para manejar solicitudes/respuestas JSON
from sklearn.feature_extraction.text import CountVectorizer  # Vectorizador para convertir texto en datos numéricos
from sklearn.linear_model import LogisticRegression  # Modelo de regresión logística para clasificación
from flask_cors import CORS  # Permite solicitudes CORS (de otros dominios) en la API
import os  # Acceso a variables de entorno y utilidades del sistema operativo
import random  # Para seleccionar consejos aleatorios
import logging  # Para registrar información y errores en la aplicación


# Configurar logging, es decir configurar el sistema de registro de eventos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configurar CORS para todos los entornos
if os.environ.get('FLASK_ENV') == 'production':
    logger.info("Ejecutándose en modo producción")
else:
    logger.info("Ejecutándose en desarrollo local")

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
    "estoy muy contento aquí", "me parece excelente la escuela",
    'Los módulos en línea son muy flexibles y me permiten trabajar al mismo tiempo, ¡es un gran apoyo!',
    'Los tutores son muy accesibles y su seguimiento personalizado me ha ayudado a no rendirme en momentos de crisis.',
    'El contenido de las asignaturas es muy pertinente y actual, lo aplico de inmediato en mi trabajo.',
    'La URC ofrece un horario a mi ritmo, lo que era la única forma de estudiar para mí.',
    'Los compañeros organizamos grupos de estudio por WhatsApp y eso nos mantiene motivados y conectados.',
    'El modelo sin costo de inscripción y colegiaturas elimina la barrera financiera más grande.',
    'Volver a estudiar a mi edad fue posible gracias a esta modalidad abierta.',
    'Los recursos bibliográficos digitales son abundantes y de fácil acceso, un punto muy a favor.',
    'La calidad de los docentes es alta, se nota su experiencia y compromiso con el modelo en línea.',
    'La flexibilidad para manejar mis propios tiempos es la razón principal por la que sigo inscrito.',
    'La oferta de licenciaturas es innovadora, me atrajo el enfoque multidisciplinario.',
    'El sistema de becas y apoyos es esencial para la permanencia de muchos estudiantes.',
    'El material audiovisual y las cápsulas informativas hacen la lectura menos pesada.',
    'La universidad está adaptada a las necesidades de la Ciudad de México y sus habitantes.',
    'El contacto constante por correo o plataforma con el tutor me da tranquilidad.',
    'Los proyectos integradores son muy útiles porque te obligan a aplicar todo lo aprendido.',
    'Ahorrar el tiempo de traslado es invaluable y me permite dedicarme a estudiar.',
    'Me siento parte de una comunidad, aunque sea virtual.',
    'La asesoría presencial es de gran ayuda para aclarar dudas específicas.',
    'El enfoque social y humanista de la universidad me motiva a seguir.',
    'He recomendado la URC porque realmente funciona para gente que trabaja.',
    'El ambiente de respeto y diversidad en los foros es muy bueno.',
    'La libertad para avanzar en los temas según mi ritmo es genial.',
    'Tuve que cambiar de carrera y el proceso fue muy ágil y me reenganchó.',
    'El costo-beneficio de estudiar en la URC es inigualable en la ciudad.',
    'La sensación de progreso al completar cada módulo corto es muy estimulante.',
    'Me gusta que los profesores sean expertos en su campo profesional.',
    'Me siento valorado por el esfuerzo que hago para combinar mi vida y el estudio.',
    'La variedad de recursos (videos, lecturas, podcasts) mantiene el interés.',
    'He desarrollado mucha disciplina y autonomía gracias a este modelo.',
    'El diseño curricular es moderno y está enfocado en las necesidades laborales actuales.',
    'Lograr compaginar mi trabajo y mis estudios es mi mayor satisfacción.',
    'Los programas de mentoría entre pares me salvaron de la deserción.',
    'Estudiar gratis es una oportunidad que no voy a desaprovechar.',
    'La flexibilidad en los exámenes (periodos largos de realización) ayuda a la planificación.',
    'El aprendizaje es significativo porque es autodirigido y yo busco la información.',
    'Siento que mi título será valorado por el enfoque práctico de la universidad.',
    'Los talleres extracurriculares en línea son un excelente complemento a la formación.',
    'Me gusta que se promueva la investigación desde el inicio de la carrera.',
    'La oportunidad de estudiar una segunda carrera sin costo es increíble.',
    'Los docentes son flexibles si se les avisa con tiempo sobre algún problema personal.',
    'Siento que estoy invirtiendo en mi futuro con una educación de calidad.',
    'Conocí a mi actual jefe en un evento de vinculación de la URC.',
    'La plataforma funciona muy bien cuando la universidad no está saturada por inscripciones.',
    'El enfoque en la innovación tecnológica me prepara para el mercado laboral.',
    'La comunidad virtual de mi carrera es muy solidaria y nos damos tips.',
    'La autonomía es un gran beneficio para adultos que ya tenemos responsabilidades.',
    'El sistema es inclusivo y valora el esfuerzo de quienes venimos de bajos recursos.',
    'La orientación vocacional previa me ayudó a elegir bien y evitar la deserción.',
    'Aprender a mi propio ritmo es lo que me mantiene en la carrera.',
    'Los eventos culturales y deportivos (aunque sean virtuales) me hacen sentir parte de la URC.',
    'El personal administrativo es muy amable cuando logras contactarlos.',
    'La oferta de posgrado de la URC me motiva a terminar la licenciatura.',
    'Los materiales de estudio están muy bien curados por los docentes.',
    'La universidad me abrió las puertas cuando pensé que ya no podría estudiar.',
    'El modelo es innovador y es un ejemplo de cómo debe ser la educación pública.',
    'La comunidad de exalumnos es muy activa y ofrece oportunidades laborales.',
    'El compromiso social que me inculcan en cada materia me encanta.',
    'Los cursos propedéuticos son muy útiles para nivelar a los que llevamos años sin estudiar.',
    'El ambiente de aprendizaje es colaborativo y de mucho respeto.',
    'La biblioteca virtual es un tesoro de conocimiento y recursos.',
    'Los profesores me inspiran a seguir investigando y aprendiendo.',
    'La URC es la mejor opción para quienes vivimos en la periferia de la CDMX.',
    'El sistema de evaluación continua me permite aprender de mis errores.',
    'La posibilidad de estudiar en línea desde casa me permite cuidar a mi familia.',
    'Los docentes siempre resuelven las dudas con paciencia.',
    'El prestigio que está ganando la universidad me da orgullo.',
    'El enfoque práctico de la carrera me permite adquirir habilidades duras.',
    'El apoyo psicológico y de bienestar es un gran valor.',
    'Me siento mucho más motivado que en una universidad tradicional.',
    'Los recursos gratuitos que ofrece la universidad son excelentes.',
    'La URC valora mi experiencia laboral previa y la integra al estudio.',
    'El tutor me ayudó a organizarme y eso me evitó desertar.',
    'El temario es muy completo y me da una base sólida para mi futuro.',
    'El programa de inglés gratuito es una gran herramienta para el futuro.',
    'Me encanta que la URC esté en constante evolución y mejora.',
    'Recibí un buen apoyo al solicitar mi baja temporal por razones de fuerza mayor.',
    'La comunicación con la dirección de carrera es muy fluida.',
    'Pude estudiar una carrera que jamás pensé poder pagar en una privada.',
    'La oferta de seminarios y congresos mantiene el estudio interesante.',
    'Logré equilibrar mi vida laboral y estudiantil gracias a la URC.',
    'El modelo educativo me reta constantemente a ser mejor y más autónomo.',
    'La asesoría académica en temas complejos es de excelente nivel.',
    'La URC me motivó a seguir estudiando después de varios años de pausa.',
    'Los grupos de apoyo de alumnos son clave para compartir experiencias y no sentirse solo.',
    'Siento un compromiso real de la institución con mi aprendizaje.',
    'Las pláticas de orientación para recién ingresados deberían ser obligatorias.',
    'El acceso a la educación es un derecho que la URC cumple cabalmente.',
    'El enfoque práctico en la resolución de problemas es genial.',
    'La flexibilidad horaria me permite tomar clases desde cualquier lugar del mundo.',
    'La formación en habilidades digitales es excelente y muy útil.',
    'La universidad es un faro de educación pública innovadora.',
    'Me siento agradecido por la oportunidad que me dio la URC.',
    'El modelo es ideal para quienes necesitan estudiar a su propio ritmo.',
    'Pude terminar mi preparatoria y luego entrar directo a la URC, ¡un sueño!',
    'El ambiente de respeto entre la comunidad estudiantil es notable.',
    'La certificación de mis conocimientos previos me ahorró tiempo valioso.',
    'Estoy orgulloso de estudiar en una institución que promueve la equidad.',
    'La metodología de estudio es muy clara y fácil de seguir.',
    'La oportunidad de colaborar en proyectos de investigación es única.',
    'El modelo me obliga a ser autodidacta y eso es una habilidad clave.',
    'Los tutores son verdaderos guías en el proceso de aprendizaje.',
    'La URC es la universidad del futuro por su modelo en línea.',
    'Me gusta la diversidad de edades y experiencias de mis compañeros.',
    'Pude validar materias de otra universidad y eso me animó a continuar.',
    'La oferta de idiomas a bajo costo es un plus increíble.',
    'La calidad de los materiales está a la altura de cualquier privada.',
    'La universidad me dio una segunda oportunidad en la vida académica.',
    'La flexibilidad de tiempos me permite ser más eficiente en el estudio.',
    'El prestigio de los convenios de la URC con otras instituciones.',
    'Las actividades de networking con egresados son muy valiosas.',
    'La posibilidad de avanzar a mi propio ritmo es ideal para mí.',
    'El sistema de créditos me permite planificar mi carga académica.',
    'Me gusta que la URC esté en la vanguardia de la educación en línea.',
    'El costo cero es la principal razón por la que muchos elegimos la URC.',
    'La universidad promueve la ética y el compromiso social.',
    'La URC es un motor de movilidad social en la CDMX.',
    'La oportunidad de reingreso después de una baja temporal fue muy fácil.',
    'Me siento feliz de haber tomado la decisión de estudiar aquí.',
    'La calidad del software educativo es sorprendente.',
    'La formación me permite ser un profesional crítico y reflexivo.',
    'Los materiales de estudio están muy bien diseñados para el autoaprendizaje.',
    'La URC es la universidad que necesitaba el futuro de la educación pública.',
    'Pude estudiar mi carrera soñada sin endeudarme, gracias a la URC.',
    'El modelo me ayuda a ser un mejor profesional y una mejor persona.',
    'El soporte técnico de la plataforma es muy eficiente y rápido.',
    'La universidad me da las herramientas para cambiar mi vida.',
    'El compromiso de los profesores con la enseñanza es visible.',
    'La oferta de prácticas profesionales es muy amplia y de calidad.',
    'Me gusta el enfoque en la innovación y la tecnología en la educación.',
    'Pude estudiar mientras viajaba por trabajo, gracias a la flexibilidad.',
    'La URC me permite ser un agente de cambio en mi comunidad.',
    'El modelo es perfecto para mí que tengo responsabilidades de cuidado.',
    'La universidad me ha dado una visión crítica del mundo.',
    'Logré mi sueño de estudiar una carrera universitaria.',
    'La URC es un orgullo para la educación pública de la ciudad.',
    'Me siento más preparado para la vida laboral que mis amigos de otras universidades.',
    'El sistema de evaluación es justo y transparente.',
    'La universidad promueve el pensamiento crítico y la autonomía.',
    'Estoy a punto de graduarme y estoy muy agradecido.',
    'La diversidad de mis compañeros enriquece mucho las discusiones.',
    'La URC es un modelo a seguir para otras universidades públicas.',
    'La formación es integral y no solo se enfoca en lo técnico.',
    'La flexibilidad me permite tomar cursos externos para complementar.',
    'Me siento apoyado por la red de tutores y compañeros.',
    'La URC me dio la esperanza de tener un futuro mejor.',
    'La universidad cumple su promesa de ser accesible y de calidad.',
    'Me siento motivado por el ejemplo de mis profesores.',
    'La oportunidad de aprender a distancia me hizo más fuerte y disciplinado.',
    'El material de apoyo de los módulos es de excelente calidad.',
    'La URC está creando líderes con conciencia social.',
    'La educación que recibo aquí es comparable con las mejores del país.',
    'La interfaz de la plataforma es muy amigable y fácil de usar.',
    'Los foros de discusión son muy activos y permiten un aprendizaje colaborativo.',
    'El enfoque práctico de las materias me ha permitido crecer profesionalmente.',
    'Agradezco la oportunidad de estudiar sin tener que desplazarme a un campus.',
    'Los webinars y sesiones en vivo son de gran calidad y muy informativos.',
    'El sistema de evaluación es justo y fomenta el autoaprendizaje.',
    'Mis profesores tienen una gran experiencia en el campo y eso se nota.',
    'La universidad se adapta a mis necesidades, no al revés.',
    'Conseguí un mejor empleo gracias a los conocimientos adquiridos aquí.',
    'La atención al estudiante es rápida y eficaz a través de los canales de soporte.',
    'El material de lectura complementario es muy útil y actualizado.',
    'Me gusta que promuevan la conciencia social y la responsabilidad.',
    'Es un modelo educativo que realmente funciona para adultos que trabajan.',
    'La posibilidad de repasar las clases grabadas es invaluable.',
    'El costo-beneficio de esta educación es inmejorable.',
    'La titulación es un proceso claro y bien acompañado.',
    'Me siento parte de una comunidad, a pesar de la distancia.',
    'La tecnología usada en la plataforma es de vanguardia.',
    'Los proyectos en equipo me han enseñan a colaborar a distancia.',
    'Es una institución que realmente cumple lo que promete.',
    'La diversidad de mis compañeros enriquece mucho las debates.',
    'Me siento menos estresado al poder manejar mis horarios.',
    'Mi tutor me inspiró a seguir una especialización.',
    'El acceso a la biblioteca virtual es muy completo.',
    'La URC es un faro de esperanza para quienes no pueden pagar una universidad privada.',
    'La curva de aprendizaje de la plataforma fue muy corta.',
    'Los exámenes son pertinentes y miden lo que se enseña.',
    'El sistema de becas y apoyos es transparente.',
    'Los convenios con empresas para prácticas son un plus.',
    'Los compañeros de mi grupo de estudio me han motivado mucho.',
    'La infraestructura tecnológica es muy estable.',
    'Me encanta la autonomía que tengo para organizar mi estudio.',
    'El enfoque humanista de la universidad me agrada.',
    'La retroalimentación de los trabajos es constructiva.',
    'Es la única forma en que mi familia me permite estudiar.',
    'La oportunidad de estudiar con gente de todo el país es genial.',
    'La certificación que obtuve tiene un buen reconocimiento.',
    'Los recursos multimedia son muy dinámicos.',
    'La orientación vocacional fue muy atinada.',
    'He desarrollado habilidades de autogestión increíbles.',
    'Siento que mi voz es escuchada en las encuestas de satisfacción.',
    'La calidad de los contenidos supera mis expectativas.',
    'Los plazos de entrega son razonables.',
    'La flexibilidad horaria es mi mayor beneficio.',
    'Los tutores son verdaderos mentores.',
    'El ambiente de respeto en los foros es ejemplar.',
    'Me siento orgulloso de estudiar en la URC.',
    'El sistema de recordatorios para las tareas es muy útil.',
    'Pude completar mi carrera sin dejar mi trabajo actual.',
    'La oferta académica está muy alineada con el mercado laboral.'
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
    "estoy perdiendo materias", "voy reprobando todo", "no puedo aprobar nada",
    'La falta de internet estable hace imposible seguir las clases a distancia, me desmotiva mucho.',
    'Sentirse aislado por la modalidad a distancia es un factor fuerte para querer dejar la carrera.',
    'La carga de trabajo es excesiva para quien tiene un empleo de tiempo completo, hay que aligerarla.',
    'La plataforma virtual a veces falla o es poco intuitiva, lo que genera frustración al entregar tareas.',
    'El proceso de reinscripción o trámites es muy lento y confuso, esto cansa y provoca abandonos.',
    'Si los apoyos económicos fueran más constantes, muchos no tendríamos que priorizar el trabajo sobre los estudios.',
    'La mala organización entre docentes y el calendario a veces nos deja tareas repetidas o sin sentido.',
    'Me siento poco preparado para el examen final porque no hubo suficientes sesiones sincrónicas para dudas.',
    'La falta de interacción presencial constante me impide desarrollar habilidades sociales clave para mi profesión.',
    'Tener que competir entre el estudio y las responsabilidades familiares me consume. Es difícil.',
    'Los foros de discusión no siempre son bien moderados o utilizados, se siente como una tarea sin valor.',
    'La burocracia interna me ha hecho perder tiempo valioso que podría usar para estudiar.',
    'No conocer a mis compañeros en persona hace que la presión de grupo positiva desaparezca.',
    'El poco dominio de herramientas tecnológicas por parte de algunos docentes dificulta la clase.',
    'Me siento abrumado por la cantidad de lecturas en poco tiempo; no hay tiempo de asimilación.',
    'La deserción por motivos de salud mental es real. Deberían tener más apoyo psicológico.',
    'Si el proceso de titulación fuera más claro desde el inicio, daría más esperanza de terminar.',
    'Los problemas personales/familiares me hicieron pausar, y la reincorporación fue muy complicada.',
    'Tener mi propia computadora y conexión a internet es un privilegio que no todos mis compañeros tienen.',
    'Los criterios de evaluación no siempre son transparentes y eso genera mucha incertidumbre.',
    'Muchos entran sin una idea clara de la disciplina y se van a los pocos meses.',
    'Me costó adaptarme a la autogestión del tiempo, casi abandono al principio.',
    'No hay suficientes espacios o centros de cómputo disponibles para quienes no tienen equipo propio.',
    'La falta de un seguimiento individual por parte del tutor, me hace sentir como un número.',
    'Las sanciones por no cumplir a tiempo son muy estrictas y no consideran imprevistos.',
    'Los problemas técnicos recurrentes son un gran obstáculo y nadie los resuelve rápido.',
    'Los trabajos en equipo virtuales son difíciles de coordinar y terminan siendo una carga desigual.',
    'La desorganización administrativa en la asignación de materias es caótica.',
    'Necesito más talleres o cursos de nivelación al inicio, sobre todo en matemáticas.',
    'Me desmotivó que un profesor me contestara groseramente en un foro público.',
    'Hay mucha desinformación sobre los requisitos de egreso, lo que genera miedo.',
    'El estrés por la autogestión me llevó a una crisis donde pensé en renunciar.',
    'La biblioteca digital a veces no tiene el libro que piden para el módulo.',
    'La frustración al no entender un tema es mayor al no tener un salón físico donde preguntar.',
    'Las constantes modificaciones al calendario escolar confunden y hacen que se pierda el ritmo.',
    'Si las fallas técnicas duran días sin solución, el tiempo perdido es irrecuperable.',
    'El trámite para hacer prácticas profesionales es una pesadilla de papeleo.',
    'No me gusta la rigidez de los formatos para los trabajos, es innecesario.',
    'La deserción se da mucho en el primer semestre porque la gente no mide el compromiso.',
    'La respuesta de las autoridades es lenta ante quejas o sugerencias.',
    'El modelo es demasiado teórico y le falta más aplicación práctica y simulaciones.',
    'La inestabilidad laboral me obliga a priorizar el trabajo y por eso me atrasé.',
    'El software que piden a veces es caro o no está disponible de forma gratuita.',
    'La falta de infraestructura para talleres presenciales en algunas licenciaturas es un freno.',
    'Me desmotiva que muchos estudiantes no se toman en serio el estudio en línea.',
    'La carga académica de verano es tan pesada que invita a dejar todo.',
    'La atención telefónica es casi nula, siempre tengo que ir en persona.',
    'El desequilibrio entre vida personal y estudio es el mayor enemigo de la permanencia.',
    'Las sesiones síncronas son obligatorias y no grabadas, lo que me perjudica por mi trabajo.',
    'El contenido de los módulos es a veces demasiado denso y no se actualiza.',
    'La poca claridad en los criterios de evaluación me genera ansiedad.',
    'Tuve que desertar por un problema de salud y el proceso de baja temporal fue muy complicado.',
    'La desconexión del mundo real al estudiar 100% en línea me afecta.',
    'Hay demasiados trámites innecesarios para algo tan simple como un justificante.',
    'La falta de seguimiento a los estudiantes en riesgo de deserción es notoria.',
    'El horario para las sesiones presenciales es inflexible y me choca con el trabajo.',
    'No hay un sistema claro para la queja o reporte de malos profesores.',
    'La deserción entre estudiantes foráneos es alta por la adaptación.',
    'Los exámenes sorpresa en el módulo final son injustos y generan estrés.',
    'La falta de recursos tecnológicos personales (tablet, PC) es una limitante.',
    'La incertidumbre sobre el valor del título de una universidad tan nueva me asusta.',
    'Me sentí perdido al pasar del bachillerato a la autogestión universitaria.',
    'El poco apoyo al emprendimiento en la carrera es decepcionante.',
    'Renuncié a mi trabajo para poder enfocarme en la carrera, fue la única forma.',
    'Me desmotivó el bajo sueldo que ofrece el campo laboral de mi carrera.',
    'Las fallas en el servidor me hicieron perder la entrega de una tarea importante.',
    'La falta de un espacio físico para estudiar o hacer tareas me afecta.',
    'Dejé la carrera por motivos financieros, aunque no se pague colegiatura, el tiempo es oro.',
    'La deserción es alta en mi generación y eso me hace dudar del programa.',
    'Las clases magistrales grabadas son demasiado largas y aburridas.',
    'La dificultad para hacer networking con profesionales externos es un problema.',
    'La poca claridad en el proceso de titulación es el mayor freno de mi generación.',
    'La falta de acompañamiento en la transición de un módulo a otro me estresó.',
    'Los requisitos de hardware y software para la carrera son muy altos y caros.',
    'La carga familiar me hizo poner en pausa la carrera indefinidamente.',
    'El desinterés de algunos compañeros en los trabajos en equipo arrastra al resto.',
    'La demora en la entrega de calificaciones genera ansiedad e incertidumbre.',
    'Desertar me pareció más fácil que seguir lidiando con problemas técnicos.',
    'El aislamiento social es real y me afecta a nivel emocional.',
    'Las reglas de permanencia son muy estrictas para un modelo flexible.',
    'La falta de seguimiento a los profesores que no cumplen es decepcionante.',
    'El descuido de mi salud mental fue la causa principal de mi abandono.',
    'No hay claridad sobre dónde buscar ayuda para problemas de la plataforma.',
    'El ritmo de los módulos es muy rápido y me siento presionada.',
    'Tener que pagar los libros a pesar de ser una universidad pública me sorprendió.',
    'La exigencia académica es muy alta, muchos no la superan.',
    'Me desanimó ver que la tasa de titulación es baja en mi carrera.',
    'El alto número de compañeros en los foros hace que mi opinión se pierda.',
    'La deserción por motivos personales (divorcio, enfermedad) no fue bien manejada administrativamente.',
    'Perdí mi empleo y eso me afectó la motivación para seguir estudiando.',
    'El sistema de tutorías es intermitente, a veces te toca un buen tutor y a veces no.',
    'El poco contacto con el mundo laboral externo durante la carrera me preocupa.',
    'La autonomía es un arma de doble filo, tienes que ser muy responsable.',
    'Los temas políticos en los foros desvían el tema central del estudio.',
    'La burocracia es un obstáculo que hay que superar, muchos se rinden ahí.',
    'Los problemas de salud son comunes en estudiantes a distancia por el sedentarismo.',
    'La falta de apoyo técnico para resolver fallas de mi equipo me frustra.',
    'La alta rotación de profesores en algunos módulos afecta la continuidad.',
    'Me sentí discriminado por mi edad en un grupo de compañeros más jóvenes.',
    'Perdí la motivación al ver que mi esfuerzo no se reflejaba en las calificaciones.',
    'Hay demasiadas interrupciones por mantenimiento en la plataforma.',
    'El costo de la electricidad y el internet es un factor de deserción para muchos.',
    'La carga mental por la autogestión me llevó al burnout.',
    'No hay suficientes espacios de apoyo para padres y madres estudiantes.',
    'El desconocimiento de las reglas de la plataforma fue mi error y me costó un módulo.',
    'La poca retroalimentación de algunos profesores hace difícil mejorar.',
    'La deserción por falta de vocación es común si no se investiga la carrera.',
    'La cantidad de spam en el correo institucional es molesta.',
    'No hay un buen balance entre la teoría y la práctica en mi licenciatura.',
    'La presión de tener que mantener buenas notas para no perder los beneficios.',
    'Me sentí desamparado al reprobar un módulo y no saber qué hacer.',
    'La falta de claridad en la misión y visión de la URC me confunde.',
    'Tuve que desertar por problemas de salud mental derivados del estrés.',
    'La desigualdad tecnológica entre estudiantes es un problema no resuelto.',
    'El ambiente virtual es muy impersonal, no se siente la comunidad universitaria.',
    'La sobrecarga administrativa a los docentes afecta la calidad de sus clases.',
    'La falta de seguimiento a los módulos reprobados es un foco de deserción.',
    'El modelo es demasiado solitario, echo de menos el ambiente de campus.',
    'Tuve que abandonar por tener que cuidar a un familiar enfermo.',
    'El desinterés de los directivos por las propuestas de mejora de los alumnos.',
    'El problema de la conexión a internet es recurrente y grave.',
    'La burocracia en el cambio de turno fue tan desesperante que me hizo pensar en irme.',
    'La desorganización en la asignación de tutores me hizo perder el primer mes.',
    'El acoso en los foros por parte de algunos compañeros no fue sancionado.',
    'La presión social y familiar por conseguir un trabajo me hizo dejar el estudio.',
    'La respuesta del área de control escolar es desesperadamente lenta.',
    'Necesito comprar libros y software que no son gratuitos, lo que afecta mi bolsillo.',
    'La calidad de algunos tutores es inconsistente, algunos no responden.',
    'La plataforma se cae justo en los días de entrega de proyectos.',
    'El proceso de revalidación de materias fue un calvario burocrático.',
    'Me siento muy solo, no hay suficientes actividades para interactuar.',
    'La falta de claridad en las rúbricas de evaluación me genera incertidumbre.',
    'Hay mucho plagio entre compañeros y no hay un castigo efectivo.',
    'El servidor de la biblioteca virtual es muy lento o a veces inaccesible.',
    'La carga administrativa es demasiada para un estudiante a distancia.',
    'El programa de becas es muy limitado, casi nadie califica.',
    'Tuve problemas para acceder a mi cuenta por un error del sistema que tardaron en resolver.',
    'No hay un seguimiento real a los estudiantes en riesgo de deserción.',
    'El material de estudio es a veces demasiado teórico y poco práctico.',
    'La falta de un espacio físico me hace extrañar el ambiente universitario.',
    'Los compañeros no se toman en serio el trabajo en equipo a distancia.',
    'La plataforma es poco accesible para personas con ciertas discapacidades.',
    'La universidad no ofrece ayuda para conseguir empleo.',
    'El proceso de reinscripción tiene errores constantes en las fechas.',
    'El sistema de mensajería interna con los profesores falla.',
    'Me sentí discriminado por mi edad en un grupo de estudio.',
    'Las actualizaciones de la plataforma siempre traen nuevos errores.',
    'Los tutores cambian a mitad del módulo sin previo aviso.',
    'La falta de contacto humano es un factor de desgaste emocional.',
    'Las videoconferencias tienen problemas de audio y video constantemente.',
    'Me desmotivó la falta de apoyo técnico para un software especializado.',
    'La comunicación entre facultades es casi nula, lo que complica los trámites.',
    'El temario es obsoleto en algunas asignaturas clave.',
    'Tuve que dejar la carrera por una emergencia familiar y la flexibilidad no fue suficiente.',
    'Los foros de debate son repetitivos y poco profundos.',
    'La cantidad de correos electrónicos de la universidad es abrumadora.',
    'Siento que la calidad de la educación ha bajado con el aumento de alumnos.',
    'La falta de orientación para el servicio social fue un problema.',
    'El plan de estudios cambió y me afectó sin previo aviso.',
    'La plataforma no está optimizada para dispositivos móviles.',
    'El sistema de quejas no es anónimo y eso da miedo.',
    'Me pidieron documentos que ya había entregado varias veces.',
    'Las evaluaciones son demasiado subjetivas en algunas materias.',
    'No hay una red de egresados activa que apoye a los nuevos.',
    'La documentación para un trámite tardó meses en ser validada.',
    'El enfoque es demasiado individualista, no fomenta la comunidad.',
    'La universidad debería ofrecer más apoyo psicológico a distancia.',
    'La lentitud en la corrección de tareas retrasa mi aprendizaje.',
    'La información en la página web no siempre coincide con la realidad.',
    'Los avisos importantes llegan tarde o no llegan.',
    'La interacción con el profesor es muy limitada.',
    'Me siento abrumado por la cantidad de lecturas obligatorias.',
    'La plataforma no guarda mi progreso a veces.',
    'El reglamento es confuso y se presta a muchas interpretaciones.',
    'Dejé de estudiar porque el estrés por las entregas era insostenible.'
]

# Combinar todos los textos de entrenamiento en una sola lista.
# Esto incluye ejemplos de matemáticas, física, química, programación, deserción, motivación,
# comentarios positivos sobre la escuela y comentarios negativos generales.
todos_los_textos = (
    textos_matematicas +
    textos_fisica +
    textos_quimica +
    textos_programacion +
    textos_desercion +
    textos_motivacion +
    textos_positivos_escuela +
    textos_negativos
)

# Crear una lista de etiquetas (categorías) correspondiente a cada texto.
# Por ejemplo, cada texto de 'textos_matematicas' recibe la etiqueta 'matematicas', etc.
todas_las_etiquetas = (
    ["matematicas"] * len(textos_matematicas) +
    ["fisica"] * len(textos_fisica) +
    ["quimica"] * len(textos_quimica) +
    ["programacion"] * len(textos_programacion) +
    ["desercion"] * len(textos_desercion) +
    ["motivacion"] * len(textos_motivacion) +
    ["positivo"] * len(textos_positivos_escuela) +
    ["negativo"] * len(textos_negativos)
)

# Convertir los textos en vectores numéricos usando CountVectorizer (bag of words).
X = vectorizer.fit_transform(todos_los_textos)
y = todas_las_etiquetas

# Entrenar el modelo de regresión logística con los datos vectorizados y sus etiquetas.
# Se usa una configuración robusta: hasta 1000 iteraciones, semilla fija para reproducibilidad,
# y parámetro C=1.0 para regularización estándar.
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(X, y)

# Imprimir en consola cuántos ejemplos y categorías se usaron para entrenar el modelo.
print(f"Modelo entrenado con {len(todos_los_textos)} ejemplos y {len(set(y))} categorías")

# Consejos profesionales con recursos específicos
consejos = {
    "matematicas": [
        "CURSOS RECOMENDADOS: Inscríbete en el curso 'Fundamentos de Álgebra' de Khan Academy (gratuito) y 'Cálculo Diferencial' en Coursera por Universidad UNAM. Contacta al Tutor de Matemáticas del Centro de Apoyo Académico de tu universidad.",
        "APOYO PERSONALIZADO: Solicita tutoría con estudiantes de Ingeniería de semestres avanzados (programa peer tutoring). El Dr. Luis García del Departamento de Matemáticas ofrece asesorías los martes y jueves de 2-4 PM.",
        "RECURSOS DIGITALES: Usa Wolfram Alpha para verificar cálculos y Photomath para resolver paso a paso. El libro 'Álgebra de Baldor' está disponible gratis en la biblioteca digital universitaria.",
        "PLAN DE ESTUDIO: Practica 30 min diarios con ejercicios graduales. Únete al Círculo de Estudio de Matemáticas que se reúne miércoles 4 PM en el aula 205. Profesora María Rodríguez coordina.",
        "CERTIFICACIÓN: Considera el curso 'Matemáticas para Ciencias' del Tecnológico de Monterrey en edX (con certificado). Te dará una base sólida y reconocimiento académico adicional."
    ],
    "fisica": [
        "LABORATORIOS ESPECIALIZADOS: Solicita acceso al Laboratorio de Física Básica fuera de horario de clase. El Ing. Roberto Martínez (Ext. 3421) coordina sesiones prácticas adicionales sábados 9-12.",
        "CURSOS COMPLEMENTARIOS: 'Física Universitaria' de UC Berkeley en edX y 'Mecánica Clásica' de MIT OpenCourseWare. Para refuerzo presencial, el Círculo de Física se reúne viernes 3 PM.",
        "RECURSOS AUDIOVISUALES: Canal de YouTube 'MinutoDeFísica' y simuladores PhET de Universidad de Colorado. La Dra. Ana López ofrece tutorías personalizadas lunes y miércoles 1-3 PM.",
        "APOYO ACADÉMICO: El programa 'Nivelación en Ciencias Exactas' del Departamento de Física ofrece cursos remediales gratuitos. Inscripciones en Servicios Escolares.",
        "PROYECTO APLICADO: Únete al Grupo de Divulgación Científica dirigido por el Dr. Carlos Hernández. Aplicar física en proyectos reales mejora comprensión y currículum."
    ],
    "quimica": [
        "LABORATORIO ABIERTO: El Laboratorio de Química General ofrece sesiones libres sábados 10-1 PM. Contacta a la Q.F.B. Patricia Morales (Ext. 2890) para reservar equipo especializado.",
        "CURSOS PROFESIONALES: 'Química General' de Universidad de Kentucky en Coursera y 'Organic Chemistry' de Yale (gratuito). Para apoyo presencial, Dr. Miguel Ángel Sánchez, cubículo 12B.",
        "SOFTWARE ESPECIALIZADO: Aprende ChemSketch (gratis) y MarvinSketch para estructura molecular. Centro de Cómputo ofrece taller 'Química Computacional' cada mes.",
        "GRUPOS DE ESTUDIO: Círculo de Química Orgánica dirigido por estudiantes de Ingeniería Química, jueves 5 PM, aula 108. Resuelven problemas complejos en equipo.",
        "VINCULACIÓN INDUSTRIAL: Participa en visitas a plantas químicas organizadas por Vinculación Universitaria. Observar procesos reales consolida conocimiento teórico."
    ],
    "programacion": [
        "CERTIFICACIONES PROFESIONALES: Google IT Automation with Python (Coursera), Microsoft Azure Fundamentals y AWS Cloud Practitioner (gratuitas para estudiantes). Centro de Vinculación Laboral te apoya.",
        "MENTORES ESPECIALIZADOS: Contacta a egresados en el programa Alumni Mentorship. El Ing. Fernando López (Google México) ofrece mentoría virtual mensual para estudiantes destacados.",
        "PROYECTOS COLABORATIVOS: Únete a los capítulos estudiantiles ACM e IEEE. Participan en hackathons y tienen biblioteca especializada en el edificio de Ingeniería.",
        "CURSOS AVANZADOS: 'CS50 Introduction to Computer Science' de Harvard (gratis), 'Python for Data Science' de IBM. El Centro de Cómputo ofrece cursos presenciales.",
        "PRÁCTICAS PROFESIONALES: Programa de Internships con empresas tech locales. Oficina de Prácticas Profesionales coordina con Microsoft, IBM y startups locales."
    ],
    "desercion": [
        "URGENTE - BIENESTAR ESTUDIANTIL: Contacta HOY al Departamento de Bienestar Estudiantil (Ext. 2500, edificio administrativo 2do piso). Psic. Laura Jiménez especializada en crisis académicas.",
        "ASESORÍA ACADÉMICA INMEDIATA: Tu Coordinador Académico puede evaluar opciones: baja temporal, reducción de materias, cambio de modalidad. Agenda cita en Servicios Escolares.",
        "APOYO FINANCIERO: Programa de Becas de Emergencia y trabajo-estudio disponible. Lic. Carmen Flores (Oficina de Becas) evalúa casos especiales. También crédito educativo FONACOT.",
        "ORIENTACIÓN VOCACIONAL: Centro de Orientación puede aplicar test vocacional y de intereses (gratuito). Confirma si tu carrera actual se alinea con tus aptitudes reales.",
        "LÍNEA DE APOYO 24/7: Universidad cuenta con línea de crisis estudiantil: 800-APOYO-U. También grupo de WhatsApp 'Red de Apoyo Estudiantil' moderado por Trabajo Social."
    ],
    "motivacion": [
        "PROGRAMA DE EXCELENCIA: Inscríbete en el Programa de Estudiantes Sobresalientes que ofrece mentorías, conferencias exclusivas y networking con profesionales exitosos.",
        "DESARROLLO DE HABILIDADES: Curso 'Técnicas de Estudio Avanzadas' del Centro de Desarrollo Estudiantil. También taller 'Liderazgo Universitario' con reconocimiento curricular.",
        "SOCIEDADES ACADÉMICAS: Únete al Consejo Estudiantil o sociedad de alumnos de tu carrera. Liderazgo estudiantil desarrolla competencias valoradas por empleadores.",
        "COACHING ACADÉMICO: Programa de coaching personalizado con egresados exitosos. 6 sesiones gratuitas enfocadas en metas académicas y desarrollo profesional.",
        "OPORTUNIDADES ESPECIALES: Postúlate a programas de intercambio, concursos académicos y proyectos de investigación con profesores. Fortalecen perfil académico significativamente."
    ],
    "positivo": [
        "EMBAJADOR ESTUDIANTIL: Con tu actitud positiva, considera ser Embajador Estudiantil para orientar a nuevos estudiantes. Programa coordinado por Relaciones Estudiantiles.",
        "ACTIVIDADES DE LIDERAZGO: Participa en el Consejo Estudiantil o comités académicos. Tu perspectiva positiva puede influir en mejoras para toda la comunidad universitaria.",
        "TESTIMONIOS INSPIRADORES: Comparte tu experiencia positiva en eventos de orientación para nuevos estudiantes. Centro de Comunicación busca historias estudiantiles exitosas.",
        "RECONOCIMIENTOS: Postúlate a programas de reconocimiento como 'Estudiante del Mes' o 'Orgullo Universitario'. Tu actitud positiva merece ser destacada institucionalmente.",
        "MENTORING: Considera ser mentor de estudiantes de primer semestre. Tu experiencia positiva puede ayudar a otros a adaptarse mejor a la vida universitaria."
    ],
    "negativo": [
        "MANEJO DEL ESTRÉS: Contacta al Centro de Bienestar Estudiantil para técnicas de manejo de estrés académico. Psic. María González ofrece talleres gratuitos de relajación martes y jueves 3-4 PM.",
        "CAMBIO DE ESTRATEGIA: Programa cita con tu Coordinador Académico para evaluar carga de materias y encontrar alternativas. A veces reducir materias mejora el rendimiento general.",
        "GRUPOS DE APOYO: Únete al Círculo de Apoyo Estudiantil que se reúne viernes 4 PM en el aula 102. Compartir experiencias con otros estudiantes alivia la presión académica.",
        "GESTIÓN DEL TIEMPO: Taller 'Organización Académica Efectiva' del Centro de Desarrollo Estudiantil, sábados 10 AM. Aprende técnicas para distribuir mejor tu carga de trabajo.",
        "TÉCNICAS DE RELAJACIÓN: App institucional 'Mindfulness UNRC' con meditaciones guiadas de 5-15 min. También yoga estudiantil gratuito en el gimnasio universitario."
    ]
}

@app.route("/clasificar", methods=["POST"])
def clasificar():
    try:
        logger.info("Recibiendo solicitud de clasificación")
        
        # Validar que la solicitud tenga datos JSON
        if not request.is_json:
            logger.error("Solicitud sin JSON válido")
            return jsonify({
                "error": "Content-Type debe ser application/json"
            }), 400
            
        data = request.get_json()
        if not data:
            logger.error("Datos JSON vacíos")
            return jsonify({
                "error": "No se recibieron datos"
            }), 400
            
        # Obtener texto, validar diferentes nombres de campo
        texto = data.get("texto") or data.get("comentario") or data.get("message", "")
        
        if not texto or not texto.strip():
            logger.error("Texto vacío recibido")
            return jsonify({
                "error": "El campo 'texto' es requerido y no puede estar vacío"
            }), 400
            
        texto = texto.lower().strip()
        logger.info(f"Procesando texto: {texto[:50]}...")
        
        # Sistema de detección mejorado con palabras clave más precisas
        palabras_muy_positivas = [
            "me gusta", "me encanta", "amo", "disfruto", "genial", "bueno", "bien",
            "contento", "feliz", "excelente", "fantástico", "maravilloso", "perfecto",
            "increíble", "fascinante", "divertido", "interesante", "motivado", "inspirado",
            "agrada", "encanta", "fascina", "maravilloso", "espectacular", "magnífico",
            "estupendo", "brillante", "admirable", "notable", "destacado", "sobresaliente",
            "positivo", "constructivo", "útil", "valioso", "beneficioso", "provechoso",
            "satisfactorio", "gratificante", "placentero", "delicioso", "gozoso",
            "alegre", "animado", "entusiasta", "apasionado", "ardiente", "ferviente",
            "emocionado", "eufórico", "exultante", "jubiloso", "radiante", "resplandeciente"
        ]

        palabras_desercion = [
            "quiero dejar", "voy a abandonar", "me quiero salir", "quiero renunciar",
            "no aguanto más", "no soporto", "odio", "abandonar", "dejar todo",
            "salirme", "cambiarme de carrera", "esto no es para mí", "renunciar",
            "abandonar estudios", "dejar universidad", "salir de la carrera"
        ]

        palabras_negativas_generales = [
            "difícil", "complicado", "no entiendo", "frustra", "estresado", "abrumado",
            "batallando", "no me sale", "me cuesta", "desespera", "fastidio", "aburrido",
            "reprobando", "reprobé", "repruebo", "calificaciones bajas", "malas notas",
            "fracasando", "fallando", "perdiendo materias", "mal en el examen", "muy mal",
            "terrible", "horrible", "pésimo", "desastroso", "catastrófico", "nefasto",
            "lamentable", "deplorable", "desagradable", "molesto", "irritante", "fastidioso",
            "tedioso", "monótono", "soporífero", "insoportable", "intolerable", "odioso",
            "detestable", "repugnante", "repulsivo", "asqueroso", "vomitivo", "nauseabundo",
            "frustrante", "desalentador", "descorazonador", "desmoralizador", "decepcionante",
            "desilusionador", "triste", "apenado", "afligido", "angustiado", "atormentado",
            "torturado", "sufrido", "doloroso", "doler", "duelo", "pena", "pesar", "congoja"
        ]

        # Frases completas que indican positividad clara sobre la institución
        frases_muy_positivas = [
            "me gusta la escuela", "me gusta estudiar", "me gusta la universidad",
            "me encanta aprender", "disfruto las clases", "amo mi carrera",
            "me gusta mi carrera", "estoy feliz estudiando", "me motiva estudiar",
            "la universidad es buena", "la escuela es buena", "estoy bien en la universidad",
            "me siento cómodo aquí", "la universidad está genial", "la escuela está genial",
            "que buena está la universidad", "me fascina mi carrera", "amo estudiar",
            "me encanta esta universidad", "disfruto mucho las clases", "me gusta aprender",
            "estoy muy contento aquí", "me parece excelente la escuela", "la universidad es excelente",
            "la escuela es excelente", "estoy feliz con mis estudios", "la universidad me gusta",
            "la escuela me gusta", "disfruto mi carrera", "amo la universidad", "amo la escuela"
        ]

        # Frases que SIEMPRE deben ser clasificadas como negativas
        frases_muy_negativas = [
            "estoy reprobando", "reprobé", "tengo malas notas", "saqué malas calificaciones",
            "me fue mal", "estoy fracasando", "no puedo aprobar", "perdí la materia",
            "tengo calificaciones bajas", "voy mal en", "me está yendo mal",
            "no logro aprobar", "siempre repruebo", "no paso las materias",
            "fracasé", "suspendí", "no aprobé", "repetí", "perdí el año",
            "estoy en riesgo de reprobar", "voy a reprobar", "tengo miedo de reprobar",
            "no entiendo nada", "no me entra nada", "esto es muy difícil para mí",
            "esto me supera", "no doy más", "estoy perdido", "no veo salida",
            "esto es imposible", "no puedo con esto", "me siento incapaz",
            "soy un fracaso", "no sirvo para esto", "esto no es para mí",
            "me equivoqué de carrera", "esta carrera no me gusta", "odio esta materia"
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
            "matematicas": "He identificado que requieres orientación en el área de matemáticas.",
            "fisica": "Tu consulta está relacionada con temas de física.",
            "quimica": "Detecto que presentas inquietudes en el ámbito de la química.",
            "programacion": "Reconozco que solicitas apoyo en programación y ciencias computacionales.",
            "desercion": "Percibo que atraviesas una situación crítica y estás considerando la posibilidad de abandonar tus estudios.",
            "motivacion": "Es positivo que busques estrategias para fortalecer tu motivación académica.",
            "positivo": "Es gratificante recibir comentarios constructivos y positivos sobre tu experiencia educativa.",
            "negativo": "Comprendo que enfrentas desafíos académicos que pueden generar estrés o frustración."
        }
        
        respuesta = respuestas_naturales.get(pred, f"He identificado tu consulta sobre: {pred}")
        
        result = {
            "respuesta": respuesta,
            "consejo": consejo,
            "categoria": pred,
            "confianza": round(confianza, 4),
            "status": "success"
        }
        
        logger.info(f"Clasificación exitosa: {pred}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error en clasificación: {str(e)}")
        return jsonify({
            "error": "Error interno del servidor",
            "message": "Ocurrió un error al procesar tu solicitud",
            "respuesta": "He recibido tu consulta académica.",
            "consejo": "Te recomiendo consultar con tu profesor o tutor académico.",
            "status": "error"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    logger.info(f"Iniciando servidor en puerto {port}")
    logger.info(f"Modo debug: {debug_mode}")
    
    if os.environ.get('FLASK_ENV') == 'production':
        logger.info("Ejecutándose en producción")
    else:
        logger.info("Ejecutándose localmente")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)