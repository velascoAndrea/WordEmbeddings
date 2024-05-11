# Trabajo de Tesis Escarleth Andrea Velasco Campos

Carne: 201503378

Para la ejecucion de los experimentos y tambiene el sistema Q&A es necesario configurar y gestionar múltiples entornos de desarrollo Python utilizando `pipenv`, una herramienta que simplifica el manejo de paquetes y entornos virtuales.

## Índice

- [Trabajo de Tesis Escarleth Andrea Velasco Campos](#trabajo-de-tesis-escarleth-andrea-velasco-campos)
  - [Índice](#índice)
  - [Pre-requisitos](#pre-requisitos)
  - [Instalación de Python 3.8](#instalación-de-python-38)
    - [Windows](#windows)
    - [macOS](#macos)
    - [Linux](#linux)
  - [Instalación de `pipenv`](#instalación-de-pipenv)
  - [Configurar un nuevo proyecto con `pipenv`](#configurar-un-nuevo-proyecto-con-pipenv)
  - [Activar el entorno virtual de `pipenv`](#activar-el-entorno-virtual-de-pipenv)
  - [Instalar dependencias desde un archivo `requirements.txt`](#instalar-dependencias-desde-un-archivo-requirementstxt)
  - [Descarga de Modelos pre-entredos de Glove y Word2vec](#descarga-de-modelos-pre-entredos-de-glove-y-word2vec)
  - [Obtener API Key de OpenAI](#obtener-api-key-de-openai)
    - [Paso 1: Crear una cuenta en OpenAI](#paso-1-crear-una-cuenta-en-openai)
    - [Paso 2: Configurar tu cuenta para acceso API](#paso-2-configurar-tu-cuenta-para-acceso-api)
    - [Paso 3: Crear una nueva API Key](#paso-3-crear-una-nueva-api-key)
    - [Paso 4: Copiar y guardar tu API Key](#paso-4-copiar-y-guardar-tu-api-key)
  - [Ejecucion de Experimentos](#ejecucion-de-experimentos)
    - [Experimento de Analogías](#experimento-de-analogías)
    - [Experimento de Vecinos Cercanos](#experimento-de-vecinos-cercanos)
    - [Experimento de Similitud Semántica](#experimento-de-similitud-semántica)
  - [Ejecucion de Sistema Q\&A](#ejecucion-de-sistema-qa)
  - [Desactivar el entorno virtual](#desactivar-el-entorno-virtual)
  - [Eliminar un entorno virtual](#eliminar-un-entorno-virtual)
  - [Consejos útiles](#consejos-útiles)

## Pre-requisitos

Antes de comenzar, asegúrate de tener Python 3.8 instalado en tu sistema, esta es la version que se utilizo durante los experimentos por lo que si se quiere obtener resultados similares es recomendable utilizar esta version. Puedes verificarlo ejecutando:

```bash
python --version
```

## Instalación de Python 3.8

Si Python no está instalado o deseas instalar Python 3.8, sigue estos pasos según tu sistema operativo:

### Windows

1. Descarga el instalador de Python 3.8 desde [python.org.](https://www.python.org/downloads/release/python-380/)
2. Ejecuta el instalador. Asegúrate de marcar la opción "Add Python 3.8 to PATH" al inicio del proceso de instalación.
3. Sigue las instrucciones en pantalla para completar la instalación.

### macOS

1. Puedes instalar Python 3.8 usando Homebrew (un gestor de paquetes para macOS). Si no tienes Homebrew, puedes instalarlo desde [brew.sh.](https://brew.sh/)
2. Abre una terminal y ejecuta: ```bash brew install python@3.8```
3. Asegúrate de que la versión correcta está configurada por defecto: ```bash brew link python@3.8 --force```

### Linux

1. La mayoría de las distribuciones de Linux permiten instalar Python directamente desde el repositorio de paquetes de la distribución. Por ejemplo, en Ubuntu puedes usar: ```bash sudo apt update sudo apt install python3.8```
2. Verifica la instalación ejecutando:  ```bash python3.8 --version```

## Instalación de `pipenv`

Para instalar pipenv, ejecuta el siguiente comando en tu terminal:

```bash
pip install pipenv
```

## Configurar un nuevo proyecto con `pipenv`

Para iniciar un nuevo proyecto y crear un entorno virtual automáticamente, navega al directorio de tu proyecto y ejecuta:

```bash
pipenv --python 3.8
```

## Activar el entorno virtual de `pipenv`

Para activar el entorno virtual generado por pipenv, utiliza el comando:

```bash
pipenv shell --python 3.8
```

## Instalar dependencias desde un archivo `requirements.txt`

Dentro de la carpeta AlgoritmosEmebeddings se encontrara un archivo llamado requirements.txt con las dependencias necesarias para la ejecucion de los experimentos, asi como tambien dentro de la carpeta SistemaQyA se encuentra un archivo requirements.txt para instalar las dependencias necesarias para el sistema se recomienda que para los experimentos y para el sistema Q&A se creen 2 entornos diferentes.

Puedes instalar todas las dependencias especificadas en él archivo con:

```bash
pipenv install -r requirements.txt
```

## Descarga de Modelos pre-entredos de Glove y Word2vec

Para poder realizar los experimentos debera descargar los archivos de los vectores pre entrenados de Glove y word2vec
[Archivos de Modelos pre-entrenados](https://drive.google.com/drive/folders/1FPbP812DJNUaq0BkuBDyKym4AQDp1RwW?usp=sharing)

## Obtener API Key de OpenAI

Para acceder a los servicios de OpenAI, como la API de GPT, necesitarás obtener una API Key.

### Paso 1: Crear una cuenta en OpenAI

1. Visita el sitio web de OpenAI [openai.com](https://www.openai.com).
2. Haz clic en "Sign Up" para registrarte.
3. Completa el formulario de registro y verifica tu cuenta.

### Paso 2: Configurar tu cuenta para acceso API

1. Una vez que hayas iniciado sesión en OpenAI, navega a la sección "API" en el menú principal.
2. Haz clic en "View API Keys".
3. Aquí podrás ver tus API Keys existentes o crear una nueva.

### Paso 3: Crear una nueva API Key

1. Haz clic en el botón "Create new key".
2. Asigna un nombre y una descripción a tu clave para identificarla fácilmente.
3. Haz clic en "Create" y tu nueva API Key será generada.

### Paso 4: Copiar y guardar tu API Key

1. Una vez creada, asegúrate de copiar tu API Key y guárdala en un lugar seguro.
2. Nunca compartas tu API Key públicamente, ya que permite acceso a tu cuenta de OpenAI y a los límites de uso asociados.

Al seguir estos pasos, podrás obtener y empezar a utilizar tu API Key de OpenAI para acceder a los servicios avanzados de inteligencia artificial que ofrece la plataforma.

## Ejecucion de Experimentos

Antes de comenzar a ejecutar experimentos, asegúrate de haber instalado todas las dependencias necesarias como se indicó en la sección [Instalar dependencias desde un archivo `requirements.txt`](#instalar-dependencias-desde-un-archivo-requirementstxt). Además, debes haber configurado correctamente las rutas a los archivos de los modelos pre-entrenados (como Word2Vec o GloVe) que deberían haber sido descargados previamente, para el caso de los experimentos de GPT debera crear un archivo llamado .env con una variable de entorno llamado OPENAI_API_KEY donde debera colocar su API KEY de OpenAI

Una vez configurado todo, puedes proceder a ejecutar los distintos experimentos con los comandos de Python. Aquí te mostramos cómo hacerlo para diferentes tipos de experimentos:

### Experimento de Analogías

1. Para ejecutar el experimento de analogías usando GloVe, utiliza el siguiente comando en tu terminal:

```bash
python3 analogiesGlove.py
```

2. Para ejecutar el experimento de analogías usando Word2Vec, utiliza el siguiente comando en tu terminal:

```bash
python3 analogiesWor2vec.py
```

3. Para ejecutar el experimento de analogías usando GPT, utiliza el siguiente comando en tu terminal:

```bash
python3 analogiesGPT.py
```

### Experimento de Vecinos Cercanos

1. Para explorar los vecinos cercanos en el espacio vectorial del modelo GloVe, ejecuta:

```bash
python3 analogiesGlove.py
```

2. Para explorar los vecinos cercanos en el espacio vectorial del modelo Word2Vec, ejecuta:

```bash
python3 analogiesWor2vec.py
```

3. Para explorar los vecinos cercanos en el espacio vectorial del modelo GPT, ejecuta:
   
```bash
python3 neighborsGPT.py
```

### Experimento de Similitud Semántica

1. Para explorar la similitud semántica en el espacio vectorial del modelo GloVe, ejecuta:

```bash
python3 similarityGlove.py
```

2. Para explorar la similitud semántica en el espacio vectorial del modelo Word2Vec, ejecuta:

```bash
python3 similarityWor2vec.py
```

3. Para explorar la similitud semántica en el espacio vectorial del modelo GPT, ejecuta:
   
```bash
python3 similarityGPT.py
```

## Ejecucion de Sistema Q&A

## Desactivar el entorno virtual

Para salir del entorno virtual de pipenv, simplemente cierra la shell o teclea `exit`.

## Eliminar un entorno virtual

Si deseas eliminar el entorno virtual y todas las dependencias asociadas al proyecto, ejecuta:

```bash
pipenv --rm
```

## Consejos útiles

- Utiliza `pipenv lock` para generar un `Pipfile.lock`, lo cual es útil para crear entornos de producción consistentes.
- Puedes usar `pipenv graph` para ver un gráfico de tus dependencias instaladas y sus relaciones.
- Asegúrate de que el entorno virtual esté activado antes de ejecutar tu código para asegurarte de que se utilizan las dependencias correctas.
