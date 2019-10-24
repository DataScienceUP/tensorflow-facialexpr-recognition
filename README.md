# Facial expression recognition in tensorflow 
This is an implementation of a convolutional neural network for facial expression recognition in tf 1.x. Code includes summaries visualization in tensorboard

All code is commented by line and consist in several parts, you have to follow the next recommendations in order to run it:

- It needs the first original data from the competition https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- You have to follow the relabeling from microsoft FER2013 plus, https://github.com/microsoft/FERPlus, reanotations script is the same as microsofts!.
- Parse and preprocess the data in such a way that you have a list of dictionaries.
- Train with landmarks or without them, download the lardmark model from `shape_predictor_68_face_landmarks.dat`


# Description

Pipeline uses tf.dataset API to load TFRecords, tensorboard and additional information that is fed to the architecture in the form of landmarks
