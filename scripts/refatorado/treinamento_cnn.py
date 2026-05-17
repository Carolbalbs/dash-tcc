
import os
import csv
import numpy as np
from scipy import stats
from PIL import Image

#from skimage import data, color
#from skimage.transform import rescale, resize, downscale_local_mean

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import regularizers, applications
 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import errno
import time as t
from datetime import datetime

print('#######################################################################################################################')
print('Script para Classificacao CNN. Data de execucao:', datetime.now())
print('#######################################################################################################################')
 
tempoInicial = t.time()
np.random.seed(2)
# ===================================================
# Configuração
# ===================================================
optimizer_name = "adam"   # <- troque aqui para mudar o otimizador
img_width, img_height = 299, 299
img_channel = 3

batch_size = 32
nb_epoch = 25

K = 5
l_r = 0.0001

# ===================================================
# Carregamento do dataset
# ===================================================
X, Y, classes, n = load_dataset(
    "/shared/rna-cnn/dataset/treinamento/output-augumented-perClass",
    299,
    shuffle=True
)
 
# ===================================================
# Diretórios de saída
# ===================================================
date_time   = datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
name        = os.path.join("./", "Resultado", f"Treinamento_{optimizer_name.upper()}_{model_name}_{date_time}")
path_img    = os.path.join(name, "imgs")
path_models = os.path.join(name, "models")
path_tb     = os.path.join(name, "logs_tensorboard")
 
for path in [path_img, path_models, path_tb]:
    os.makedirs(path, exist_ok=True)
 
print(f"Diretório de resultados: {name}")
 
# ===================================================
# CSV de resultados
# ===================================================
summary_csv_path = os.path.join(name, f"results_{optimizer_name.upper()}_{model_name}.csv")
print(f"CSV de resultados: {summary_csv_path}")

def load_dataset(base_dir,img_size,shuffle=True):
    X = []
    Y = []
    processed_image_count = 0
    skipped = 0
    classes = sorted([
    dir_path for dir_path in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir,dir_path))
])
    print(f"Classes Encontradas:({len(classes)}):(classes)")

    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()
            if extension not in VALID_EXTENSIONS:
                skipped +=1
                continue
            file_path = os.path.join(root, filename)

            suffix = file_path[len(base_dir):].lstrip(os.sep)
            label = suffix.slipt(os.sep)[0]

            if label not in classes:
                skipped +=1
                continue

            img = Image.open(file_path).convert('RGB')
            img = np.array(img.resize((img_size,img_size)), dtype='float32')
            img = img / 255.

            X.append(img)
            Y.append(classes.index(label))

            processed_image_count+=1

    print(f"Imagens carregadas: {processed_image_count}")
    print(f"Arquivos ignorados: {skipped}")

    X = np.array(X, dtype='float32')
    Y = np.array(Y)

    print("Shuffle:"+str(shuffle))
    if shuffle:
        perm = np.random.permutation(len(Y))
        X = X[perm]
        Y = Y[perm]
    return X, Y, classes, processed_image_count
X, Y, classes, n = load_dataset("",299,True)

def callbacks(index):
    callback = [
        ModelCheckpoint(
            os.path.join(path_models, f"best_modelFold{i}.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max', 
            verbose=1
        ),
        CSVLogger(
            os.path.join(name, f"log_Fold{index}.csv"),
            append=True,
            separator=';'
        ),
        TensorBoard(
            log_dir=os.path.join(path_tb, f"Fold{index}"),
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
    ]
    return callback

def create_model(optmizer_):
    bn_momentum = 0.4 
    l2 = 0.0001
    l_r = 0.001

    base_model = InceptionV3(
        include_top=False,
        weigths=None,
        pooling='max',
        input_shape=(img_width, img_height, img_channel)
    )
    add_model = Sequential([
        Dense(1024, activation='relu',input_shape=base_model.output_shape[1:]),
        Dropout(0.6),
        Dense(2, activation='softmax')
    ])
    model = Model(
       inputs=base_model.input,
       outputs=add_model(base_model.output)
   )
    if optimizer_ == "sgd":
        optimizer = SGD(learning_rate=l_r)
    elif optimizer_ == "adam":
        optimizer = Adam(learning_rate=l_r)
    elif optimizer_ == "rmsprop":
        optimizer = RMSprop(learning_rate=l_r)
    else:
        raise ValueError(f"Otimizador '{optimizer_}' inválido. Use: sgd, adam ou rmsprop")

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model

def load_model_and_return_score(path, xval, yval):
    print ("****************" + path)
    model = load_model(path)
    scores = model.evaluate(xval, yval, verbose=1)
    return scores[1] * 100

def save_results(cvscores, classes, n_images, tempo_minutos):

    n        = len(cvscores)
    mean     = np.mean(cvscores)
    std      = np.std(cvscores)
    ic       = stats.t.interval(0.95, df=n-1, loc=mean, scale=stats.sem(cvscores))

    with open(summary_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')

        # Cabeçalho geral
        writer.writerow(['Classes',       ', '.join(classes)])
        writer.writerow(['Total imagens', n_images])
        writer.writerow(['Otimizador',    optimizer_name.upper()])
        writer.writerow(['Modelo',        model_name])
        writer.writerow(['', ''])

        # Resultados por fold
        writer.writerow(['Fold', 'Accuracy (%)'])
        for i, score in enumerate(cvscores):
            writer.writerow([f'Fold {i+1}', f'{score:.2f}'])

        writer.writerow(['', ''])

        # Estatísticas finais com t-Student
        writer.writerow(['Métrica',              'Valor'])
        writer.writerow(['Média Accuracy',        f'{mean:.2f}%'])
        writer.writerow(['Desvio Padrão',         f'{std:.2f}%'])
        writer.writerow(['IC 95% Inferior',       f'{ic[0]:.2f}%'])
        writer.writerow(['IC 95% Superior',       f'{ic[1]:.2f}%'])
        writer.writerow(['Tempo de execução (min)', f'{tempo_minutos:.2f}'])

    print(f"\nResultados salvos em: {summary_csv_path}")
    print(f"Média: {mean:.2f}% | Desvp: {std:.2f}% | IC 95%: [{ic[0]:.2f}%, {ic[1]:.2f}%]")

def save_plots(history, index):
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acuracia')
    plt.ylabel('Acuracia')
    plt.xlabel('Epoca')
    plt.legend(['Treino', 'Teste'], loc='upper left')
    fig.savefig(os.path.join(path_img, 'acc_Fold' + str(index) + '.png'), bbox_inches='tight', dpi=300)
    plt.close();

# ===================================================
# Validacao cruzada
# ===================================================
cvscores = []
skf = StratifiedKFold(n_splits=K, shuffle=True)
 
for index, (train_indices, val_indices) in enumerate(skf.split(X, Y)):
    print(f'\n{"_"*70}')
    print(f"Treinando fold {index + 1}/{K}...")
 
    xtrain, xval = X[train_indices], X[val_indices]
    ytrain, yval = Y[train_indices], Y[val_indices]
 
    ytrain = to_categorical(ytrain, len(classes))
    yval   = to_categorical(yval,   len(classes))
 
    print(f"Treino: {xtrain.shape[0]} imagens | Validação: {xval.shape[0]} imagens")
 
    model     = create_model(optimizer_name)
    callbacks = callbacks(index)
 
    history = model.fit(
        xtrain, ytrain,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(xval, yval),
        callbacks=callbacks,
        verbose=1
    )
 
    # Salva modelo final do fold
    pathModelFile = os.path.join(path_models, f"modelFold{index}.keras")
    model.save(pathModelFile)
 
    # Avalia o MELHOR modelo salvo pelo ModelCheckpoint
    best_model_path = os.path.join(path_models, f"best_modelFold{index}.keras")
    best_model      = load_model(best_model_path)
    scores          = best_model.evaluate(xval, yval, verbose=1)
    cvscores.append(scores[1] * 100)
 
    save_plots(history, index)
 
    print(f'{"_"*70}')
    print(f"Fold {index + 1} concluído")
    print(f"Modelo final  : {pathModelFile}")
    print(f"Melhor modelo : {best_model_path}")
    print(f"Accuracy      : {scores[1]*100:.2f}%")
 
# ===================================================
# Resultados finais
# ===================================================
tempoFim          = t.time()
tempoEmMinutos    = (tempoFim - tempoInicial) / 60
 
save_results(cvscores, classes, n, tempoEmMinutos)
 
print('_______________________________________________________________________')
print('Fim de Treinamento')
print(f'Tempo de execução: {tempoEmMinutos:.2f} minuto(s).')
print('_______________________________________________________________________\n')