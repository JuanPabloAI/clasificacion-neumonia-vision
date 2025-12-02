import json
import os

notebook_path = 'notebooks/03_clasificacion_con_descriptores_clasicos.ipynb'

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb['cells']

    # 1. Insert MinMaxScaler
    # Find the cell with StandardScaler
    scaler_index = -1
    for i, cell in enumerate(cells):
        if "source" in cell:
            source = "".join(cell["source"])
            if "StandardScaler" in source and "fit_transform" in source:
                scaler_index = i
                break

    if scaler_index != -1:
        minmax_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "minmax_scaler",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Normalización con MinMaxScaler (0-1)\n",
                "minmax_scaler = MinMaxScaler()\n",
                "X_train_minmax = minmax_scaler.fit_transform(X_train_clean)\n",
                "X_test_minmax = minmax_scaler.transform(X_test_clean)\n",
                "X_val_minmax = minmax_scaler.transform(X_val_clean)\n",
                "\n",
                "print(\"\\n✅ Datos normalizados con MinMaxScaler\")\n",
                "print(f\"Min del train minmax: {X_train_minmax.min():.6f}\")\n",
                "print(f\"Max del train minmax: {X_train_minmax.max():.6f}\")"
            ]
        }
        # Check if already exists to avoid duplicates if run multiple times
        if "minmax_scaler" not in "".join(cells[scaler_index+1].get("source", [])):
             cells.insert(scaler_index + 1, minmax_cell)
             print("Added MinMaxScaler cell.")
        else:
             print("MinMaxScaler cell already exists.")

    # 2. Insert SelectKBest
    # Find the cell with PCA visualization
    pca_index = -1
    for i, cell in enumerate(cells):
        if "source" in cell:
            source = "".join(cell["source"])
            if "PCA para visualización" in source:
                pca_index = i
                break

    if pca_index != -1:
        selectkbest_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "selectkbest",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Selección de características con SelectKBest\n",
                "k_best = 500  # Seleccionar las 500 mejores características\n",
                "selector = SelectKBest(f_classif, k=k_best)\n",
                "X_train_selected = selector.fit_transform(X_train_scaled, y_train)\n",
                "X_test_selected = selector.transform(X_test_scaled)\n",
                "X_val_selected = selector.transform(X_val_scaled)\n",
                "\n",
                "print(f\"\\n✅ Selección de características completada\")\n",
                "print(f\"Forma original: {X_train_scaled.shape}\")\n",
                "print(f\"Forma reducida: {X_train_selected.shape}\")\n",
                "\n",
                "# Obtener índices de las características seleccionadas\n",
                "selected_indices = selector.get_support(indices=True)\n",
                "print(f\"Índices de las primeras 10 características seleccionadas: {selected_indices[:10]}\")"
            ]
        }
        if "SelectKBest" not in "".join(cells[pca_index+1].get("source", [])):
            cells.insert(pca_index + 1, selectkbest_cell)
            print("Added SelectKBest cell.")
        else:
            print("SelectKBest cell already exists.")

    # 3. Append CNN
    # Check if CNN already exists
    cnn_exists = False
    for cell in cells:
        if "source" in cell and "Clasificación con Redes Neuronales Convolucionales" in "".join(cell["source"]):
            cnn_exists = True
            break
    
    if not cnn_exists:
        cnn_header_cell = {
            "cell_type": "markdown",
            "id": "cnn_header",
            "metadata": {},
            "source": [
                "### 7. Clasificación con Redes Neuronales Convolucionales (CNN)\n",
                "Implementamos una CNN básica utilizando las imágenes en crudo."
            ]
        }

        cnn_code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": "cnn_impl",
            "metadata": {},
            "outputs": [],
            "source": [
                "import tensorflow as tf\n",
                "from tensorflow.keras.models import Sequential\n",
                "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
                "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
                "\n",
                "# Configuración\n",
                "IMG_WIDTH, IMG_HEIGHT = 150, 150\n",
                "BATCH_SIZE = 32\n",
                "EPOCHS = 5\n",
                "TRAIN_DIR = '../data/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/train'\n",
                "TEST_DIR = '../data/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test'\n",
                "\n",
                "print(\"🔄 Preparando generadores de datos...\")\n",
                "\n",
                "# Generadores de datos con reescalado\n",
                "train_datagen = ImageDataGenerator(rescale=1./255)\n",
                "test_datagen = ImageDataGenerator(rescale=1./255)\n",
                "\n",
                "train_generator = train_datagen.flow_from_directory(\n",
                "    TRAIN_DIR,\n",
                "    target_size=(IMG_WIDTH, IMG_HEIGHT),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode='binary'\n",
                ")\n",
                "\n",
                "test_generator = test_datagen.flow_from_directory(\n",
                "    TEST_DIR,\n",
                "    target_size=(IMG_WIDTH, IMG_HEIGHT),\n",
                "    batch_size=BATCH_SIZE,\n",
                "    class_mode='binary'\n",
                ")\n",
                "\n",
                "# Definir modelo CNN\n",
                "model = Sequential([\n",
                "    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),\n",
                "    MaxPooling2D(2, 2),\n",
                "    Conv2D(64, (3, 3), activation='relu'),\n",
                "    MaxPooling2D(2, 2),\n",
                "    Conv2D(128, (3, 3), activation='relu'),\n",
                "    MaxPooling2D(2, 2),\n",
                "    Flatten(),\n",
                "    Dense(512, activation='relu'),\n",
                "    Dropout(0.5),\n",
                "    Dense(1, activation='sigmoid')\n",
                "])\n",
                "\n",
                "model.compile(optimizer='adam',\n",
                "              loss='binary_crossentropy',\n",
                "              metrics=['accuracy'])\n",
                "\n",
                "print(\"\\n🚀 Entrenando CNN...\")\n",
                "history = model.fit(\n",
                "    train_generator,\n",
                "    steps_per_epoch=train_generator.samples // BATCH_SIZE,\n",
                "    epochs=EPOCHS,\n",
                "    validation_data=test_generator,\n",
                "    validation_steps=test_generator.samples // BATCH_SIZE\n",
                ")\n",
                "\n",
                "print(\"\\n✅ Entrenamiento completado\")\n",
                "test_loss, test_acc = model.evaluate(test_generator, verbose=2)\n",
                "print(f'\\nAccuracy en test: {test_acc:.4f}')"
            ]
        }

        cells.append(cnn_header_cell)
        cells.append(cnn_code_cell)
        print("Appended CNN cells.")
    else:
        print("CNN cells already exist.")

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
