
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm


seq_length = 40
class_limit = None  

data = DataSet(seq_length=seq_length, class_limit=class_limit)

model = Extractor()

pbar = tqdm(total=len(data.data))
newdata=data.data[:2]
for video in data.data:

    path = os.path.join('data', 'sequences', video[2] + '-' + str(seq_length) + \
        '-features')  

    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue


    frames = data.get_frames_for_sample(video)
    print(frames)


    frames = data.rescale_list(frames, seq_length)

    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    np.save(path, sequence)

    pbar.update(1)

pbar.close()
