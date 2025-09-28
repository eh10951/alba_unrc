# Asistente Escolar Alba - UNRC

Una aplicación web de asistente virtual para estudiantes universitarios que utiliza reconocimiento de voz y machine learning para clasificar consultas académicas.

## 🚀 Características

- **Reconocimiento de voz**: Interfaz de voz en español
- **Clasificación inteligente**: Usa machine learning para categorizar consultas
- **Interfaz moderna**: Diseño estilo Jarvis con efectos visuales
- **Respuestas personalizadas**: Consejos específicos según el tipo de consulta

## 🛠️ Tecnologías

- **Frontend**: HTML5, CSS3, JavaScript (Web Speech API)
- **Backend**: Python, Flask, scikit-learn
- **Machine Learning**: Clasificación de texto con TF-IDF

## 📁 Estructura del proyecto

```
my_ml_app_vos/
├── frontend/
│   ├── index.html          # Interfaz web principal
│   ├── mascota.jpeg        # Imagen de la mascota
│   └── mascota3.jpeg       # Imagen alternativa
├── backend/
│   └── app.py             # Servidor Flask y modelo ML
└── minijarvis/            # Entorno virtual Python
```

## 🚀 Instalación y uso

### Requisitos previos
- Python 3.8+
- Navegador web moderno con soporte para Web Speech API

### Configuración

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

2. Crea un entorno virtual:
```bash
python -m venv venv
venv\Scripts\activate  # En Windows
```

3. Instala las dependencias:
```bash
pip install flask flask-cors scikit-learn numpy
```

4. Ejecuta el servidor backend:
```bash
cd backend
python app.py
```

5. Abre `frontend/index.html` en tu navegador

## 💡 Uso

1. Haz clic en "🎤 Habla con Alba"
2. Permite el acceso al micrófono
3. Habla tu consulta académica
4. Alba clasificará tu consulta y te dará una respuesta personalizada

## 🎓 Universidad

Desarrollado para la Universidad Rosario Castellanos (UNRC)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.