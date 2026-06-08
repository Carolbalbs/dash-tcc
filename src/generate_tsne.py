import numpy as np
import pandas as pd
import os
try:
    from tf_keras.models import load_model, Model
    from tf_keras.preprocessing.image import img_to_array, load_img
    print("Using tf_keras for model loading.")
except ImportError:
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    print("Using tensorflow.keras for model loading.")
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import pickle

# Configuration
CSV_PATH = "data/dimensoes-dataset/image_dimensions_dataset_novo.csv"
MODEL_PATH = "data/Resultado_dataset-antigo/Treinamento_ADAM_InceptionV3_03-05-2026__18-08-53/models/best_modelFold3.keras"
OUTPUT_DIR = "data/trained_data_odontogenic"
IMG_SIZE = 150

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# The model has InceptionV3 as the first main component after Input
# In the script: base_model = InceptionV3(..., pooling='max')
# add_model = Sequential([Dense(1024), Dropout(0.6), Dense(2)])
# feature_extractor should be the model up to the layer before the final dense layers.
# Let's find the layer that outputs the 2048-dim vector (pooling='max' from InceptionV3)
feature_layer_name = None
for layer in model.layers:
    if "inception_v3" in layer.name:
        feature_extractor = layer
        break
else:
    # If it's a flat model where Inception layers are directly in the model
    # we look for the global_max_pooling2d layer
    for layer in model.layers:
        if "global_max_pooling2d" in layer.name or "max_pooling2d" in layer.name:
             feature_layer_name = layer.name

if feature_layer_name:
    feature_model = Model(inputs=model.input, outputs=model.get_layer(feature_layer_name).output)
else:
    # Fallback: take the layer before the first Dense layer in the top part
    # Or just use the model as is if it's already a feature extractor
    # In the user's script, model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    # So model.layers[-1] is the Sequential 'add_model'
    # and model.layers[-2] is the 'base_model' (InceptionV3)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

print("Feature model summary:")
feature_model.summary()

# 2. Load images and extract features
df = pd.read_csv(CSV_PATH)
# Filter to ensure we have the files (some might be missing or on different paths)
# We'll use the 'path' column from the CSV, but we might need to adjust it if it's absolute and incorrect for this env.
# The CSV has /home/carol/shared/rna-cnn/dataset/originais/Axiais_QO/IM-0001-00141.png

features = []
labels = []
valid_paths = []

print(f"Processing {len(df)} images...")
for i, row in df.iterrows():
    img_path = row['path']
    if not os.path.exists(img_path):
        # Try relative path if absolute fails
        rel_path = os.path.join("shared", row['path'].split("/shared/")[-1])
        if os.path.exists(rel_path):
            img_path = rel_path
        else:
            continue
    
    try:
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        
        feat = feature_model.predict(x)
        features.append(feat.flatten())
        
        # Mapping: Axiais_DO -> 1 (Dentigero), Axiais_QO -> 0 (Queratocisto) 
        # based on Prediction.py analysis
        label = 0 if "Axiais_QO" in row['directory'] else 1
        labels.append(label)
        valid_paths.append(img_path)
        
        if len(features) % 50 == 0:
            print(f"Processed {len(features)} images...")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

features = np.array(features)
labels = np.array(labels)

print(f"Extracted features shape: {features.shape}")

# 3. Run T-SNE
print("Running T-SNE...")
tsne = TSNE(n_components=2, random_state=42)
embeddings = tsne.fit_transform(features)

# 4. Train Linear Model for new embeddings (for the upload feature)
print("Training linear model for embedding approximation...")
linear_model = LinearRegression()
linear_model.fit(features, embeddings)

# 5. Save everything
print(f"Saving data to {OUTPUT_DIR}...")
np.save(os.path.join(OUTPUT_DIR, "all_images_tsne.npy"), embeddings)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
with open(os.path.join(OUTPUT_DIR, "image_paths.pkl"), "wb") as f:
    pickle.dump(valid_paths, f)
with open(os.path.join(OUTPUT_DIR, "linear_model_embeddings.sav"), "wb") as f:
    pickle.dump(linear_model, f)

print("Done!")
