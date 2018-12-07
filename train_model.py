from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data import DataSet
import os.path

data = DataSet()

#Saving the Model
checkpointer = ModelCheckpoint(
    filepath=os.path.join('data', 'savedmodels', 'inception.model.hdf5'),
    verbose=1,
    save_best_only=True)

#Stop after 10 epochs when there is no progress in Learning
early_stopper = EarlyStopping(patience=10)

tensorboard = TensorBoard(log_dir=os.path.join('data', 'logs'))

def get_generators():
    
    #Image Data Preperation(Generates batches of tensor image data with real-time data augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)
    
    #Image Data Preperation(Generates batches of tensor image data with real-time data augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    #Takes the path to a directory & generates batches of augmented data.
    train_generator = train_datagen.flow_from_directory(
        os.path.join('data', 'frames/train'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    #Takes the path to a directory & generates batches of augmented data.
    validation_generator = test_datagen.flow_from_directory(
        os.path.join('data', 'frames/test'),
        target_size=(299, 299),
        batch_size=32,
        classes=data.classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model(weights='imagenet'):
    
    #Pre Trained base Model
    base_model = InceptionV3(weights=weights, include_top=False)
   
    # Add a global spatial average pooling layer
    x = base_model.output
    
    
    x = GlobalAveragePooling2D()(x)
 
    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
  
    # Now add a logistic layer for classification.
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # The Model that we train
    model = Model(inputs=base_model.input, outputs=predictions)
 
    #Compile The Model.
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
   

    return model

#Train just the Top Layers of the Model
def freeze_all_but_top(model):
    
    #Train only Top Layers of the Model and
    #Freeze all the convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False

    #Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model):

    #Train top 2 inception blocks which is freeze 172 blocks and unfreeze the rest.
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    #We should recompile the model for the modifications to happen
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    print("train model start")
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        validation_data=validation_generator,
        validation_steps=5,
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
        #Train Top Layers
        model = freeze_all_but_top(model)
        model = train_model(model, 10, generators)
        print("3.if-> train done")
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)
        print("3.else-> load done")
    #Train Mid layers
    model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 10, generators,
                        [checkpointer, early_stopper, tensorboard])
    print("4.final model done")

if __name__ == '__main__':
    weights_file = None
    main(weights_file)
