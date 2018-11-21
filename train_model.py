from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data import DataSet
import os.path

data = DataSet()

checkpointer = ModelCheckpoint(
    # filepath=os.path.join('data', 'savedmodels', 'inception.{epoch:03d}-{val_loss:.2f}.hdf5'),
    filepath=os.path.join('data', 'savedmodels', 'inception.model.hdf5'),
    verbose=1,
    save_best_only=True)

early_stopper = EarlyStopping(patience=10)

tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)
    print("gg1")

    test_datagen = ImageDataGenerator(rescale=1./255)
    print("gg2")

    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'frames/train'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
    print("gg3")

    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'frames/test'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')
    print("gg4")

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    # create the base pre-trained model
    base_model = InceptionV3(weights=weights, include_top=False)
    print("gm1")

    # add a global spatial average pooling layer
    x = base_model.output
    print("gm2")
    
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    print("gm3")
    
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    print("gm4")
    
    predictions = Dense(len(data.classes), activation='softmax')(x)
    print("gm5")

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print("gm6")
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print("gm7")

    return model


def train_model(model, nb_epoch, generators, callbacks=[]):
    print("train model start")
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=1,
        validation_data=validation_generator,
        validation_steps=1,
        epochs=nb_epoch,
        callbacks=callbacks)
    print("train model end")
    return model

def main(weights_file):
    model = get_model()
    print("1.model done")
    generators = get_generators()
    print("2.generators done")
    if weights_file is None:
        print("Loading network from ImageNet weights.")
        model = train_model(model, 1, generators)
        print("3.if-> train done")
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)
        print("3.else-> load done")

    model = train_model(model, 1, generators,
                        [checkpointer, early_stopper, tensorboard])
    print("4.final model done")

if __name__ == '__main__':
    weights_file = None
    main(weights_file)
