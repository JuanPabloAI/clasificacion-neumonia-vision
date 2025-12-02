# 🫁 Clasificación de Neumonía con Visión por Computador

**Trabajo 3 - Visión por Computador**  
**Universidad Nacional de Colombia - Facultad de Minas**

## 📋 Descripción del Proyecto

Este proyecto implementa y compara sistemas de clasificación de imágenes médicas (radiografías de tórax) para detectar neumonía utilizando dos enfoques:

1. **Descriptores Clásicos** (*handcrafted features*) + Clasificadores tradicionales (SVM, Random Forest, k-NN, Logistic Regression)
2. **Deep Learning** con Redes Neuronales Convolucionales (CNNs) - *Parte 4 (opcional)*

## 🎯 Objetivos

- Explorar y preprocesar un dataset médico de radiografías
- Implementar descriptores de forma y textura desde conceptos de visión por computador
- Entrenar y evaluar clasificadores tradicionales
- Comparar rendimiento entre diferentes enfoques
- Documentar el proceso completo en un pipeline reproducible

## 📊 Dataset

**Chest X-Ray Pneumonia Detection**  
[Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Train**: ~5,216 imágenes (1,341 NORMAL | 3,875 PNEUMONIA)
- **Test**: ~624 imágenes
- **Validation**: ~16 imágenes
- **Formato**: JPEG, dimensiones variables
- **Desbalance**: 3:1 (Pneumonia:Normal)

## 🏗️ Estructura del Proyecto

```
clasificacion-neumonia-vision/
├── README.md                                    # Este archivo
├── requirements.txt                             # Dependencias del proyecto
├── Trabajo03.md                                 # Enunciado del trabajo
├── data/                                        # Datasets (descargados automáticamente)
│   └── datasets/
│       └── paultimothymooney/
│           └── chest-xray-pneumonia/
│               └── versions/2/chest_xray/
│                   ├── train/
│                   ├── test/
│                   └── val/
├── notebooks/                                   # Notebooks de Jupyter
│   ├── 01_exploracion_y_preprocesamiento.ipynb  # Análisis EDA + CLAHE
│   ├── 02_extraccion_de_descriptores.ipynb      # HOG, LBP, GLCM, Gabor, etc.
│   ├── 03_clasificacion_con_descriptores_clasicos.ipynb  # SVM, RF, k-NN, LR, CNN.
└── results/                                     # Resultados generados
    ├── features_classical.npz                   # Características extraídas
    └── figures/                                 # Visualizaciones
```

## 🛠️ Instalación y Configuración

### Requisitos Previos

- **Python 3.10+** (recomendado: 3.11 o 3.12)
- **Anaconda/Miniconda** (opcional pero recomendado)
- **Cuenta de Kaggle** (para descarga automática del dataset)

### Configuración en macOS y Linux

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/clasificacion-neumonia-vision.git
cd clasificacion-neumonia-vision

# 2. Crear entorno virtual
python3 -m venv .venv

# 3. Activar entorno
source .venv/bin/activate

# 4. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configurar Jupyter (si no está instalado globalmente)
python -m ipykernel install --user --name=.venv --display-name "Python (Pneumonia)"

# 6. Lanzar Jupyter
jupyter notebook
```

### Configuración en Windows

```cmd
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/clasificacion-neumonia-vision.git
cd clasificacion-neumonia-vision

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno
.venv\Scripts\activate

# 4. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configurar Jupyter
python -m ipykernel install --user --name=.venv --display-name "Python (Pneumonia)"

# 6. Lanzar Jupyter
jupyter notebook
```

### Configuración de Kaggle API (Descarga Automática)

El dataset se descarga automáticamente al ejecutar los notebooks. Para que funcione:

1. **Crear cuenta en Kaggle**: [kaggle.com](https://www.kaggle.com)
2. **Generar API Token**:
   - Ir a: Account → API → Create New API Token
   - Se descarga `kaggle.json`
3. **Configurar credenciales**:
   ```bash
   # macOS/Linux
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   
   # Windows
   mkdir %USERPROFILE%\.kaggle
   copy kaggle.json %USERPROFILE%\.kaggle\
   ```

## 🚀 Uso del Proyecto

### Generar Figuras para el Blog Post

Para generar todas las visualizaciones necesarias para el blog post de GitHub Pages:

```bash
# Ejecutar manualmente desde Jupyter
jupyter notebook
# Luego ejecutar todos los notebooks en orden
```

**Figuras generadas**:
- `01_muestras_radiografias.png` — Ejemplos de imágenes NORMAL y PNEUMONIA
- `02_distribucion_clases.png` — Distribución del dataset
- `03_comparacion_clahe.png` — Comparación entre CLAHE y ecualización estándar
- `04_hog_visualization.png` — Visualización del descriptor HOG
- `05_lbp_visualization.png` — Visualización del descriptor LBP
- `06_gabor_filters.png` — Banco de filtros de Gabor
- `07_feature_distributions.png` — Distribuciones de características discriminativas
- `08_top_discriminative_features.png` — Top 20 características (Cohen’s d)
- `09_pca_2d_visualization.png` — Proyección PCA en 2D
- `10_pca_variance_explained.png` — Varianza explicada por cada componente principal
- `11_cv_comparison.png` — Comparación de desempeño en validación cruzada
- `12_metrics_comparison.png` — Comparación global de métricas de los modelos
- `13_confusion_matrices.png` — Matrices de confusión de **todos los modelos clásicos**
- `14_roc_curves.png` — Curvas ROC comparadas
- `15_feature_importance.png` — Importancia de características según Random Forest
- `16_confusion_matrix_cnn.png` — Matriz de confusión del modelo CNN
- `17_roc_cnn.png` — Curva **ROC** del modelo CNN

### Copiar Figuras a GitHub Pages

```bash
# Crear carpeta de assets
mkdir -p docs/assets/images

# Copiar todas las figuras
cp results/figures/*.png docs/assets/images/

# Verificar
ls docs/assets/images/
```

### Orden de Ejecución de Notebooks

1. **`01_exploracion_y_preprocesamiento.ipynb`**
   - Descarga automática del dataset
   - Análisis de distribución de clases
   - Implementación de CLAHE (mejora de contraste)
   - Visualización de preprocesamiento

2. **`02_extraccion_de_descriptores.ipynb`**
   - **Descriptores de Forma**:
     - HOG (Histogram of Oriented Gradients)
     - Momentos de Hu (7 invariantes)
     - Descriptores de Contorno (área, perímetro, circularidad, excentricidad)
   - **Descriptores de Textura**:
     - LBP (Local Binary Patterns)
     - GLCM (Gray Level Co-occurrence Matrix)
     - Filtros de Gabor
     - Estadísticas de Primer Orden
   - Construcción de matriz de características
   - Guardado de features en `results/features_classical.npz`

3. **`03_clasificacion_con_descriptores_clasicos.ipynb`**
   - Normalización con StandardScaler
   - Reducción de dimensionalidad (PCA)
   - Entrenamiento de clasificadores:
     - Logistic Regression
     - SVM (Linear y RBF)
     - Random Forest
     - k-NN
   - Validación cruzada estratificada (5-Fold)
   - Entrenamiento de CNN
   - Evaluación con métricas:
     - Accuracy, Precision, Recall, F1-Score
     - Matrices de Confusión
     - Curvas ROC y AUC
   - Análisis de importancia de características
   - Optimización de hiperparámetros (GridSearchCV)

## 📈 Resultados Esperados

### Métricas Clave

- **Accuracy**: Precisión general del modelo
- **Recall (Sensibilidad)**: **CRÍTICO** - Minimizar falsos negativos (no detectar neumonía)
- **F1-Score**: Balance entre precisión y recall
- **AUC-ROC**: Capacidad discriminativa del modelo

### Baseline de Referencia

Debido al desbalance (74.3% Pneumonia), un clasificador que siempre prediga "Pneumonia" obtendría:
- **Accuracy Base**: 74.3%
- **Recall Base**: 100% (pero con muchos falsos positivos)

**Meta**: Superar significativamente este baseline con modelos entrenados.

## 🧪 Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.10+ |
| **Procesamiento de Imágenes** | OpenCV, scikit-image |
| **Machine Learning** | scikit-learn, scipy, tensorflow |
| **Visualización** | matplotlib, seaborn |
| **Notebooks** | Jupyter, IPython |
| **Gestión de Datos** | NumPy, pandas |
| **Descarga de Datasets** | kagglehub |

## 📚 Conceptos Implementados

### Preprocesamiento
- ✅ Normalización de tamaño (224x224)
- ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- ✅ Binarización con Otsu

### Descriptores de Forma
- ✅ **HOG**: Detecta bordes y estructuras (costillas, clavículas)
- ✅ **Momentos de Hu**: Invariantes a traslación, escala y rotación
- ✅ **Contornos**: Caracterización geométrica de regiones pulmonares

### Descriptores de Textura
- ✅ **LBP**: Patrones locales de textura
- ✅ **GLCM**: Relaciones espaciales entre píxeles
- ✅ **Gabor**: Filtros direccionales multi-frecuencia
- ✅ **Estadísticas**: Media, varianza, skewness, kurtosis, entropía

### Clasificadores
- ✅ SVM (kernels linear y RBF)
- ✅ Random Forest (con análisis de importancia)
- ✅ k-NN (vecinos cercanos)
- ✅ Logistic Regression
- ✅ Convolutional Neural Networks

### Evaluación
- ✅ Validación cruzada estratificada
- ✅ Métricas robustas al desbalance
- ✅ Matrices de confusión
- ✅ Curvas ROC
- ✅ Optimización de hiperparámetros

## 🔧 Solución de Problemas

### Error: `ValueError` en histogramas (NumPy 2.2.6)

**Problema**: Incompatibilidad entre NumPy 2.2.6 y matplotlib.

**Solución aplicada**: Uso de `np.bincount()` en lugar de `np.histogram()`.

### Dataset no se descarga

**Causas**:
1. No hay credenciales de Kaggle configuradas
2. Red bloqueando Kaggle API

**Solución**:
```bash
# Verificar configuración
cat ~/.kaggle/kaggle.json

# Descargar manualmente desde: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Descomprimir en: data/datasets/paultimothymooney/chest-xray-pneumonia/
```

### Kernel de Jupyter no aparece

```bash
python -m ipykernel install --user --name=.venv --display-name "Python (Pneumonia)"
jupyter kernelspec list  # Verificar instalación
```

## 👥 Equipo

**Grupo:** Grillo Digital  
**Integrantes:**
- Juan Pablo Palacio Pérez - [juppalaciope@unal.edu.co](mailto:juppalaciope@unal.edu.co)
- David Giraldo Valencia - [dgiraldova@unal.edu.co](mailto:dgiraldova@unal.edu.co)
- Andrés Felipe Moreno Calle - [amorenocal@unal.edu.co](mailto:amorenocal@unal.edu.co)
- Víctor Manuel Velásquez Cabeza - [vivelasquezc@unal.edu.co](mailto:vivelasquezc@unal.edu.co)

## 📖 Referencias

1. Kermany, D. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell.
2. Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
3. Ojala, T. et al. (2002). *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*. PAMI.
4. Haralick, R.M. et al. (1973). *Textural Features for Image Classification*. IEEE Transactions on Systems, Man, and Cybernetics.
5. [PyImageSearch - Hu Moments](https://pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/)

## 📄 Licencia

Este proyecto es parte de un trabajo académico para la Universidad Nacional de Colombia.

## 🙏 Agradecimientos

- Profesor: Juan David Ospina Arango
- Monitor: Andrés Mauricio Zapata
- Dataset: Paul Mooney (Kaggle)

---

**Última actualización**: Diciembre 2025
