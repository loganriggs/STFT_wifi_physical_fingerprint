import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
#Ignore UserWarning from using complex values for STFT
warnings.filterwarnings("ignore")
from utils import Augment_complex, stft_converter

#Parameters
#augment
num_augments = 10 #* Affects runtime
augment_method = 1  #1 adds random signal, 0 adds random vector
aug_std_dev=5.0e-3

#STFT nperseg (number of samples) parameters
# frequencies = [10, 20, 40, 60, 80,100, 120, 140, 160, 180]
frequencies = [5, 10, 20, 30, 50, 100, 200]
# frequencies = [10,20, 40, 60, 80]
frequencyLoop = len(frequencies)

#Loop Parameters
averageRange = 10 #* Affects runtime

#Models to run ("mlp", "cnn")
modelToRun = "mlp"

#Results storage
accuracyArray = np.zeros((averageRange, 1))
meanArray = np.zeros((frequencyLoop,1))
stdArray = np.zeros((frequencyLoop,1))
maxArray = np.zeros((frequencyLoop,1))
minArray = np.zeros((frequencyLoop,1))
freq = 1
count = 0

#CNN Parameters
epochs = 50 #* Affects runtime
batchSize = 128 #* Affects runtime
opt = 'adam'
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#MLP model Parameters
epochs = 1000  #MLP runs until converges or until this number of iterations
model_size = (64,64,64,64)
alpha = 1e-4 #Learning Rate

#Load Dataset in
file_prefix = "classification_"
x_train  = np.load(file_prefix + "x_train.npy")
y_train = np.load(file_prefix + "y_train.npy")
x_test = np.load(file_prefix + "x_test.npy")
y_test = np.load(file_prefix + "y_test.npy")
classes = max([max(y_train), max(y_test)])+1

#Augment only Training Data
print("x_train shape Original: ", x_train.shape)
# x_train, y_train = Augment_complex(x_train, y_train, num_augments, augment_method=augment_method, aug_std_dev=aug_std_dev)
print("x_train shape Augmented: ", x_train.shape)



for n in range(frequencyLoop):
    print("Sampling Size: ", n)
    nPerSeg = frequencies[n]

    #Convert to STFT
    freqDataTrain, inputSize = stft_converter(data= x_train, sampling_nperseg=nPerSeg)
    freqDataTest, inputSize = stft_converter(data= x_test, sampling_nperseg=nPerSeg)

    #Run the model N times for average, variance, and max
    for index in range(averageRange):

        if(modelToRun == "cnn"):
            #TODO: Dr. Ball, here is the model
            model = models.Sequential()
            model.add(layers.Conv2D(32, (5, 5), activation='relu', padding="same", input_shape=inputSize + (1,)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=0.3))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(rate=0.3))
            model.add(layers.MaxPooling2D((2, 2)))
            # model.add(layers.Conv2D(64, (5, 5), padding="same", activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.Dense(classes))

            model.compile(optimizer=opt,
                          loss=loss,
                          metrics=['accuracy'])

            history = model.fit(freqDataTrain, y_train, epochs=epochs, batch_size = batchSize,
                                validation_data = (freqDataTest, y_test),
                               verbose = 1)
            accuracyArray[index] = history.history['val_accuracy'][-1]
            del history, model
        elif(modelToRun == "mlp"):
            clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=model_size,
                                max_iter=epochs)
            clf.fit(np.abs(freqDataTrain), y_train)
            y_hat = clf.predict(np.abs(freqDataTest))
            accuracy = accuracy_score(y_test, y_hat)
            accuracyArray[index] = accuracy
            print("Accuracy MLP-Normal: " + str(accuracy))
    #Calculate mean, std, max, min for this frequency
    meanArray[n] = np.mean(accuracyArray)
    stdArray[n] = np.std(accuracyArray)
    maxArray[n] = np.max(accuracyArray)
    minArray[n] = np.min(accuracyArray)
    #
    # print("Freq: ", nPerSeg)
    # print("Mean: ", meanArray)
    # print("Std: ", stdArray)
    # print("Max: ", maxArray)
print("Freq: ", frequencies)
print("Mean: ", meanArray)
print("Std: ", stdArray)
print("Max: ", maxArray)

for f, m in zip(frequencies, maxArray):
    print(f, ", ",m)

#Saving Data
np.savetxt("accuraciesMean" + modelToRun + ".txt", meanArray)
np.savetxt("accuraciesStd" + modelToRun + ".txt", stdArray)
np.savetxt("accuraciesMax" + modelToRun + ".txt", maxArray)
np.savetxt("accuraciesMin" + modelToRun + ".txt", minArray)

print(len(frequencies), " == ", meanArray.shape)
fig, ax = plt.subplots()
plt.errorbar(frequencies, meanArray, stdArray, fmt='ok', lw=3, capsize=3)
plt.errorbar(frequencies, meanArray, [meanArray-minArray, maxArray-meanArray],
             fmt='.k', ecolor='gray', lw=1, capsize=3)
plt.xlabel('Size of Sampling Window')
plt.ylabel('Accuracy')
plt.xticks(frequencies)
plt.title('Mean and Standard Deviation of STFT Accuracy')
# Save the figure and show
plt.ylim((0.0,1))
plt.tight_layout()
plt.show()