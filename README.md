# NeuralNetworksProject - VIDEO CLASSIFICATION
To run the program, please follow the instructions given below:
1) Download the UCF-101 dataset and extract it in the data folder.
2) Run the preprocessing.py file of the data folder. This file will split your dataset into train and test data. To run the file, type 'python preprocessing.py' on your terminal.
3) After running preprocessing.py, run preprocessing2.py as: 'python preprocessing2.py'. preprocessing2.py will generate and store all the frames for each video of every category.
4) Make three directories by typing the command: 'mkdir checkpoints logs sequences'
5) Go to the previous directory.
6) To run method-1 (as mentioned in the report), run train_model.py by typing the following command in the terminal: 'python train_model.py'. To test the model, type the following command: 'test_model.py'
7) To run method-2 (as mentioned in the report), run train_lrcn.py by typing the following command in the terminal: 'python train_lrcn.py'. To test this model, type the following command: 'test_lrcn.py'
