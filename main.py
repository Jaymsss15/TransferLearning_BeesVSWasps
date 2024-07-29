import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pyswarm import pso
from itertools import product
import random
import matplotlib.pyplot as plt
import itertools
import pandas as pd

# Parâmetros e Configurações
im_shape = (299, 299)
TRAINING_DIR = 'C:/Users/Utilizador/Desktop/Faculdade/Universidade/IC/kaggle_bee_vs_wasp/train'
TEST_DIR = 'C:/Users/Utilizador/Desktop/Faculdade/Universidade/IC/kaggle_bee_vs_wasp/test'
seed = 10
BATCH_SIZE = 200
BEST_AC = 0
FLAG = 0

# Configuração do ImageDataGenerator para aumento de dados
data_generator = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

# Carregamento do conjunto de dados usando geradores de fluxo
train_generator = data_generator.flow_from_directory(
    TRAINING_DIR, target_size=im_shape, shuffle=True, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE, subset="training"
)
validation_generator = val_data_generator.flow_from_directory(
    TRAINING_DIR, target_size=im_shape, shuffle=False, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE, subset="validation"
)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    TEST_DIR, target_size=im_shape, shuffle=False, seed=seed,
    class_mode='categorical', batch_size=BATCH_SIZE
)

# Número de classes e lista de classes
num_classes = 3
classes = ['bees', 'wasps', 'other_insects']

# Função para treinamento e avaliação do modelo
def train_evaluate_model(dropout_rate, learning_rate, train_generator, validation_generator, test_generator):
    print(f"Training model with Dropout Rate: {dropout_rate}, Learning Rate: {learning_rate}")
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(im_shape[0], im_shape[1], 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(50, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    base_model.trainable = False
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=1,
        validation_data=validation_generator,
        verbose=1,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Avaliação no conjunto de teste
    score = model.evaluate_generator(test_generator)
    test_loss, test_accuracy = score[0], score[1]
    print(f"Test accuracy: {test_accuracy}")

    return test_loss, test_accuracy, model

# PSO
def pso_optimization(params):
    dropout_rate, learning_rate = params
    test_loss, test_accuracy, _ = train_evaluate_model(dropout_rate, learning_rate, train_generator, validation_generator, test_generator)
    BEST_AC = test_accuracy;
    FLAG = 1;
    return -test_accuracy  # Minimizar a métrica negativa (negativo da acurácia)

# Definir limites para os hiperparâmetros
lower_bound = [0.0, 1e-5]  # lower bound para dropout e learning rate
upper_bound = [0.5, 1e-2]  # upper bound para dropout e learning rate
bounds = (lower_bound, upper_bound)

# Executar PSO para otimizar os hiperparâmetros
print("Running PSO optimization...")
best_params_pso, _ = pso(pso_optimization, lb=bounds[0], ub=bounds[1], swarmsize=1, maxiter=1)

# Exibir os melhores parâmetros encontrados pelo PSO
print("Best Dropout Rate (PSO):", best_params_pso[0])
print("Best Learning Rate (PSO):", best_params_pso[1])

# Grid Search
dropout_values = [0.3]
learning_rate_values = [1e-3]
best_val_accuracy_grid = 0.0
best_params_grid = {}
print("Running Grid Search...")
for dropout_rate, learning_rate in product(dropout_values, learning_rate_values):
    test_loss, test_accuracy, _ = train_evaluate_model(dropout_rate, learning_rate, train_generator, validation_generator, test_generator)
    if test_accuracy > best_val_accuracy_grid:
        best_val_accuracy_grid = test_accuracy
        best_params_grid = {'dropout_rate': dropout_rate, 'learning_rate': learning_rate}

# Exibir os melhores parâmetros encontrados pelo Grid Search
print("Best Dropout Rate (Grid Search):", best_params_grid['dropout_rate'])
print("Best Learning Rate (Grid Search):", best_params_grid['learning_rate'])
print("Best Test Accuracy (Grid Search):", best_val_accuracy_grid)

if best_val_accuracy_grid > BEST_AC:
    BEST_AC = best_val_accuracy_grid;
    FLAG = 2;

# Random Search
num_random_search_iterations = 1
best_val_accuracy_random = 0.0
best_params_random = {}
print("Running Random Search...")
for _ in range(num_random_search_iterations):
    dropout_rate = random.choice(dropout_values)
    learning_rate = random.uniform(1e-4, 1e-2)
    test_loss, test_accuracy, _ = train_evaluate_model(dropout_rate, learning_rate, train_generator, validation_generator, test_generator)
    if test_accuracy > best_val_accuracy_random:
        best_val_accuracy_random = test_accuracy
        best_params_random = {'dropout_rate': dropout_rate, 'learning_rate': learning_rate}

# Exibir os melhores parâmetros encontrados pelo Random Search
print("Best Dropout Rate (Random Search):", best_params_random['dropout_rate'])
print("Best Learning Rate (Random Search):", best_params_random['learning_rate'])
print("Best Test Accuracy (Random Search):", best_val_accuracy_random)

if best_val_accuracy_random > BEST_AC:
    BEST_AC = best_val_accuracy_random;
    FLAG = 3;

# Escolher os dois melhores conjuntos de hiperparâmetros
#best_params_list = [best_params_pso, best_params_grid, best_params_random]
#best_params_list.sort(key=lambda x: -train_evaluate_model(x[0], x[1], train_generator, validation_generator, test_generator)[1])
#best_params_final = best_params_list[:2]

if FLAG == 1:
    best_params = best_params_pso
elif FLAG == 2:
    best_params = best_params_grid
elif FLAG == 3:
    best_params = best_params_random
else:
    raise ValueError("Valor inválido para best_params")

# Exibir os dois melhores conjuntos de hiperparâmetros
    print(f"Dropout Rate: {best_params['dropout_rate']}, Learning Rate: {best_params['learning_rate']}")

# Treinar o modelo final com os melhores parâmetros encontrados pelo PSO
final_test_loss, final_test_accuracy, model = train_evaluate_model(best_params['dropout_rate'], best_params['learning_rate'], train_generator, validation_generator, test_generator)

# Exibir a acurácia final no conjunto de teste
print("Best Test Accuracy (Using the Two Best Sets of Hyperparameters):", final_test_accuracy)

# Avaliação do modelo nos conjuntos de validação e teste
score = model.evaluate_generator(validation_generator)
print('Val loss:', score[0])
print('Val accuracy:', score[1])

score = model.evaluate_generator(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predição no conjunto de teste e criação da matriz de confusão
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(test_generator.classes, y_pred)

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plotagem da matriz de confusão
plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix')

# Relatório de classificação
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=classes))


# Resultados do PSO
pso_results = {'Dropout Rate (PSO)': [best_params_pso[0]],
               'Learning Rate (PSO)': [best_params_pso[1]],
               'Test Accuracy (PSO)': [BEST_AC]}

# Resultados do Grid Search
grid_search_results = {'Dropout Rate (Grid Search)': [best_params_grid['dropout_rate']],
                       'Learning Rate (Grid Search)': [best_params_grid['learning_rate']],
                       'Test Accuracy (Grid Search)': [best_val_accuracy_grid]}

# Resultados do Random Search
random_search_results = {'Dropout Rate (Random Search)': [best_params_random['dropout_rate']],
                         'Learning Rate (Random Search)': [best_params_random['learning_rate']],
                         'Test Accuracy (Random Search)': [best_val_accuracy_random]}

# Criar DataFrames
pso_df = pd.DataFrame(pso_results)
grid_search_df = pd.DataFrame(grid_search_results)
random_search_df = pd.DataFrame(random_search_results)

# Salvar os DataFrames em um arquivo Excel
excel_writer = pd.ExcelWriter('resultados.xlsx', engine='xlsxwriter')

pso_df.to_excel(excel_writer, sheet_name='PSO', index=False)
grid_search_df.to_excel(excel_writer, sheet_name='Grid_Search', index=False)
random_search_df.to_excel(excel_writer, sheet_name='Random_Search', index=False)

# Salvar a matriz de confusão
cm_df = pd.DataFrame(cm, columns=classes, index=classes)
cm_df.to_excel(excel_writer, sheet_name='Confusion_Matrix')

# Salvar o arquivo Excel
excel_writer.close()