import io
import base64
import os
import pickle
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Mapping based on generate_tsne.py logic
label_mapping = {
    0: "Queratocisto (Axiais_QO)",
    1: "Dentígero (Axiais_DO)"
}

def parse_image(contents, filename, date):
    # Take uploaded image, from dcc upload, convert to np array, and reshape for InceptionV3
    content_type, content_string = contents.split(",")
    im = Image.open(io.BytesIO(base64.b64decode(content_string)))
    im = im.convert('RGB')
    
    # Resize for InceptionV3 (299x299)
    resized_im = im.resize((299, 299), Image.LANCZOS)
    resized_arr = np.array(resized_im) / 255.0
    
    # Return original image (as PIL object or b64) and resized array
    return contents, resized_arr

def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255 if needed
    if scalar and array.max() <= 1.0:
        array = np.uint8(255 * array)
    else:
        array = np.uint8(array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return "data:image/png;base64," + im_b64

def create_img(path):
    if isinstance(path, str):
        if not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(path, np.ndarray):
        return numpy_to_b64(path)
    return ""

def load_tsne_data(data_dir="trained_data_odontogenic"):
    all_X_hat = np.load(os.path.join(data_dir, "all_images_tsne.npy"))
    all_labels = np.load(os.path.join(data_dir, "labels.npy"))
    with open(os.path.join(data_dir, "image_paths.pkl"), "rb") as f:
        image_paths = pickle.load(f)
    linear_model = pickle.load(open(os.path.join(data_dir, "linear_model_embeddings.sav"), "rb"))
    
    return all_X_hat, all_labels, image_paths, linear_model
