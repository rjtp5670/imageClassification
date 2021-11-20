# ImageClassification

- Desgined by David Park
- No License
- Contact: rjtp5670@gmail.com

This is my personal AI project to take a bite of AI world. Honestly, I have no background knowledge about AI (And have lack of programming experience, not gonna lie). So that most of codes are not written from scratch and by myself. Thanks to my tutor, Junhee, he helps me a lot to work around, but also Tensorflow Tutorial & references with Google search.

I has done my best adding very detailed explanation based on my noob experience, so that hope this help someone to understand the difficult AI world.

## Operating Enviroment

- Windows 10
- Visual Studio Code
- TensorFlow v2.6 (GPU used, Nvidia Geforce 1050)
- Python v3.8.8 (Setup using Anaconda)

### TensorFlow Installation (GPU Setup)

- TensorFlow v2.6 (GPU used, Nvidia Geforce 1050)

### Dataset Load

Concord Grape, Crimson Grape, Shine Msucat Grape, Thompson Seedless Grape are classes.

`Download`: [Grape Data Set](https://drive.google.com/file/d/1hrMcXlr-kjzr0QF7QZcmMrBv16tyCqEI/view?usp=sharing)

<img src="img/dataset_load_concord.png"  width="500">
> Concord Grapes

<img src="img/dataset_load_chrimson.png"  width="500">
> Crimson Grape

<img src="img/dataset_load_shine.png"  width="500">
> Shine Msucat Grape

<img src="img/dataset_load_thompson.png"  width="500">
> Thompson Seedless Grape

### Preprocessing - ImageDataGenerator

Keras provides Image Augumentation generator that increases dataset training.

```python
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocessing
    rotation_range=10,  # Loop within specified ranges
    zoom_range=0.1,  # Zoom images
    width_shift_range=0.1,  # Shift image
    height_shift_range=0.1,  #
    horizontal_flip=True,  # Flip Image
    vertical_flip=True,  #
    validation_split=0.2  # Use 20% images for a validation.
)
```

### Build Model

```python
def build_model(units):

    resnet = ResNet50(
        include_top=False,
        pooling="avg",
        weights="imagenet"
    )
    for layer in resnet.layers[:-10]:
        layer.trainable = False or isinstance(layer, BatchNormalization)

    logits = Dense(units)(resnet.layers[-1].output)
    output = Activation('softmax')(logits)

    model = Model(resnet.input, output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

#### Transfer Learning

When you have limited datasets, you can bring predefined training network and apply the trained feature from [`ResNet50`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50).

### Training

Load a model to train using `fit` method

```python
history = model.fit(
    train_generator,  #
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[tensorboard_callback,
                # early_stopping,
                model_checkpoint_callback]
)

units = len(class_names)
model = build_model(units=units)

```

### Confusion Matrix

Generate a predict matrix where each cell shows the counts value of training result. Therefore, you can visually see the built model performance.

<img src="img\Confusion_Matrixpng.png"  width="500">

> Confusion Matrix: The more accuracy the classifier can get, the more deeper blue it is.

```python
def show_confusion_matrix(model, validation_generator):
    validation_generator.reset()
    y_preds = model.predict(validation_generator)
    y_preds = np.argmax(y_preds, axis=1)
    y_trues = validation_generator.classes
    cm = confusion_matrix(y_trues, y_preds)

    fig, ax = plt.subplots(figsize=(7, 6))  # 700 x 600 px

    # Adding up with "True"
    # Apply color bar on heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)

    ax.set(
        xticklabels=list(label_to_class.keys()),  # X-Axis label: Prediction
        yticklabels=list(label_to_class.keys()),  # Y-Axis label: True
        title='confusion matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )

    params = dict(rotation=45, ha='center', rotation_mode='anchor')

    plt.setp(ax.get_yticklabels(), **params)
    plt.setp(ax.get_xticklabels(), **params)
    plt.show()


# Display Confusion Matrix.
show_confusion_matrix(model, validation_generator)

```

### Performing the trained model

Classify the type of grape into the right class according to the trained model justification. A test image is a thompson grape. (\input_img\thompson_0005.jpg)

```python

Grapes = r'C:\Users\hansung\Documents\GitHub\imageClassification\grape-dataset' # Doesn't have to be a path.
class_names = list(sorted(os.listdir(Grapes)))

# Dictionary Comprehenstion
class_to_label = dict([(i, class_name)
                      for i, class_name in enumerate(class_names)])

# Convert image to numpy array(4D array)
def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')  #
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

# Test Image Path. Can be any images
filename = r'C:\Users\hansung\Documents\GitHub\imageClassification\input_img\thompson_0005.jpg'
model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
# Load model .h5 file
model = tf.keras.models.load_model(model)

image = load(filename)  #

image = preprocess_input(image)
y_pred = model(image)[0]
print(y_pred)
cls = np.argmax(y_pred)
print(f'{class_to_label[cls]}: {(y_pred[cls]*100):.1f}%')
print([f'{label}: {(pred*100):.1f}' for label,
      pred in zip(class_to_label.values(), y_pred)])
pred_img = Image.open(filename).resize((256, 256))
pred_img.show()
```
