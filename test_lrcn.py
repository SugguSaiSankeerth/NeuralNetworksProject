
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from lrcn import ResearchModels
from data import DataSet

def validate(data_type, model, seq_length=40, saved_model=None,
             class_limit=None, image_shape=None):
    batch_size = 32

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

    val_generator = data.frame_generator(batch_size, 'test', data_type)

    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=3200)

    print(results)
    print(rm.model.metrics_names)

def main():
    model = 'lrcn'
    saved_model = 'data/checkpoints/lrcn-features.026-0.239.hdf5'

    if model == 'conv_3d' or model == 'lrcn':
        data_type = 'images'
        image_shape = (80, 80, 3)
    else:
        data_type = 'features'
        image_shape = None

    validate(data_type, model, saved_model=saved_model,
             image_shape=image_shape, class_limit=4)

if __name__ == '__main__':
    main()
