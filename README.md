# ImageClassification

- Desgined by David Park
- No License
- Contact: rjtp5670@gmail.com

This is my personal AI project to take a bite of AI world. Honestly, I have no background knowledge about AI (And have lack of programming experience, not gonna lie). So that most of codes are not written from scratch and by myself. Thanks to my tutor, Junhee, he helps me a lot to work around, but also Tensorflow Tutorial & references with Google search.

I has done my best adding very detailed explanation based on my noob experience, so that hope this help someone to understand the difficult AI world.

<!--

Grape Data Set Download Link

https://drive.google.com/file/d/1hrMcXlr-kjzr0QF7QZcmMrBv16tyCqEI/view?usp=sharing

-->

## Operating Enviroment

- Windows 10
- Visual Studio Code
- TensorFlow v2.6 (GPU used, Nvidia Geforce 1050)
- Python v3.8.8 (Setup using Anaconda)

### TensorFlow Installation (GPU Setup)

- TensorFlow v2.6 (GPU used, Nvidia Geforce 1050)

### Dataset Load

- Classes: Concord Grape, Crimson Grape, Shine Msucat Grape, Thompson Seedless Grape.

Download the [Grape Data Set](https://drive.google.com/file/d/1hrMcXlr-kjzr0QF7QZcmMrBv16tyCqEI/view?usp=sharing)

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

### Classification
