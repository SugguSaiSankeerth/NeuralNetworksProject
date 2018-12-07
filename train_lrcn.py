
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from lrcn import ResearchModels
from data import DataSet
import time
import os.path

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    #Save the Model
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'savedmodels', model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    #TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    #Stop after 5 epochs when there is no progress in Learning
    early_stopper = EarlyStopping(patience=5)

    #Save Results in csv format
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    #Process the Data
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    #Get Steps per epoch
    #Guess how much of data.data is Train data by multiplying with 0.7
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        #Get Data
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
          #Get Generators
          generator = data.frame_generator(batch_size, 'train', data_type)
          val_generator = data.frame_generator(batch_size, 'test', data_type)

    #Get Model
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    #Fit by using Standard fit
    if load_to_memory:
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def main():
    #Main Training Settings

    #Model can be lrcn,conv_3d,c3d,lstm,mlp
    model = 'lrcn'
    saved_model = None  #None or weights File to save this model
    class_limit = None  #int, can be 1-101 or None
    seq_length = 40
    load_to_memory = False  #pre-load the sequences into Memory
    nb_epoch = 1000

    # Chose Data Type and Image Shape based on model.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
