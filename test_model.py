
import numpy as np
import operator
import random
import glob
import os.path
from sklearn import metrics

from data import DataSet
from data import process_image
from keras.models import load_model


def main(video):

    data = DataSet()
    model = load_model('data/savedmodels/inception.model.hdf5')
    file=video[2:-8]
    print(file)

    images = glob.glob(os.path.join('data','frames', 'test',file, video, '*.jpg'))
    

    path=os.path.join('data','frames', 'test',file, video)
    print(path)
    nb_images=len(images)
    print(nb_images)
    class_label_predictions={}

    for i, label in enumerate(data.classes):
            class_label_predictions[i] = 0

    for sample in range(nb_images):
        print('-'*80)

        image = images[sample]

        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        predictions = model.predict(image_arr)

        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]
            class_label_predictions[i] += (predictions[0][i]/nb_images)


        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        
        for i, class_prediction in enumerate(sorted_lps):
            if i > 4:
                break
            print(i)
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1
    print('-'*80)
    print('-'*80)
    class_sorted_lps = sorted(class_label_predictions.items(), key=operator.itemgetter(1), reverse=True)
    for i, class_prediction in enumerate(class_sorted_lps):
        if i > 4:
            break
        print(i)
        print("%s: %.2f" % (data.classes[class_prediction[0]], class_prediction[1]))
        i += 1

    return data.classes[class_sorted_lps[0][0]]





def accuracy():
    videos=glob.glob(os.path.join('data','frames', 'test', '*','*'))

    print(os.path.basename(videos[0]))
    labels=[]
    predicted_labels=[]
    for i in  videos:
        video=os.path.basename(i)

        labels.append(video.split('_')[1])
        predicted_labels.append(main(video))


    acc=metrics.accuracy_score(labels,predicted_labels)
    print(acc)



if __name__ == '__main__':

    accuracy()
