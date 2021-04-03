from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from utils import Augment_complex, stft_converter, cnn1D, cnn2D


#Parameters
#MLP
epochs = 1000
model_size = (64,64,64,64)
alpha = 1e-4
#CNN
epochsCNN = 50 #* Affects runtime
batchSize = 128 #* Affects runtime
#STFT
overlap = 3

file_prefix = "classification_"
x_train  = np.load(file_prefix + "x_train.npy")
y_train = np.load(file_prefix + "y_train.npy")
x_test = np.load(file_prefix + "x_test.npy")
y_test = np.load(file_prefix + "y_test.npy")
classes = max([max(y_train), max(y_test)])+1
print("Number of Classes: ", classes)

#Networks to run
mlpNormal=True
mlpSTFT = True
mlpAugSTFT = True
mlpConcSTFT = True
cnnNormal=False
cnnSTFT = False
cnnAugSTFT = False
cnnConcSTFT = False

if(mlpAugSTFT or cnnAugSTFT):
    #Augment the data (Train only)
    num_augments = 2
    augment_method = 1
    aug_std_dev=5.0e-3
    x_train_aug, y_train_aug = Augment_complex(x_train, y_train, num_augments, augment_method=augment_method, aug_std_dev=aug_std_dev)
else:
    x_train_aug, y_train_aug =  x_train, y_train
#cnnData
cnn1D_x_train = np.expand_dims(x_train, axis=2)
cnn1D_x_test = np.expand_dims(x_test, axis=2)


#STFT Parameters
nPerSeg = 200  #frequency bins



# #MLP Normal
if(mlpNormal):
    clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=model_size, random_state=1, max_iter=epochs)
    clf.fit(np.abs(x_train), y_train)
    y_hat = clf.predict(np.abs(x_test))
    print("Accuracy MLP-Normal: " + str(accuracy_score(y_test, y_hat)))

if(cnnNormal):
    inputSize = (x_train.shape[1],1)
    print("xtrain shape: ", cnn1D_x_train.shape)
    print("input Size cnn: ", inputSize)
    model = cnn1D(inputSize, classes=classes)
    history = model.fit(cnn1D_x_train, y_train, epochs=epochsCNN, batch_size=batchSize,
                        validation_data=(cnn1D_x_test, y_test),
                        verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    print("Accuracy CNN1D-Normal: " + str(accuracy))

if(mlpSTFT):
    freqDataTrain, _ = stft_converter(data= x_train, sampling_nperseg=nPerSeg, overlap=overlap)
    freqDataTest, _ = stft_converter(data= x_test, sampling_nperseg=nPerSeg, overlap=overlap)

    # MLP STFT Flatten
    clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=model_size, random_state=1, max_iter=epochs)
    clf.fit(np.abs(freqDataTrain), y_train)
    y_hat = clf.predict(np.abs(freqDataTest))
    print("Accuracy MLP-STFT: " + str(accuracy_score(y_test, y_hat)))

if(cnnSTFT):
    freqDataTrain, inputSize = stft_converter(data= x_train, sampling_nperseg=nPerSeg, setting="cnn", overlap=overlap)
    freqDataTest, inputSize = stft_converter(data= x_test, sampling_nperseg=nPerSeg, setting="cnn", overlap=overlap)

    model = cnn2D(inputSize, classes=classes)
    history = model.fit(freqDataTrain, y_train, epochs=epochsCNN, batch_size=batchSize,
                        validation_data=(freqDataTest, y_test),
                        verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    print("Accuracy CNN2D-STFT: " + str(accuracy))



if(mlpAugSTFT):
    freqDataTrain, _ = stft_converter(data= x_train_aug, sampling_nperseg=nPerSeg, overlap=overlap)
    freqDataTest, _ = stft_converter(data= x_test, sampling_nperseg=nPerSeg, overlap=overlap)

    # MLP STFT Flatten
    clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=model_size, random_state=1, max_iter=epochs)
    clf.fit(np.abs(freqDataTrain), y_train_aug)
    y_hat = clf.predict(np.abs((freqDataTest)))
    print("Accuracy MLP-Augmented-STFT: " + str(accuracy_score(y_test, y_hat)))

if(cnnAugSTFT):
    freqDataTrain, inputSize = stft_converter(data=x_train_aug, sampling_nperseg=nPerSeg, setting="cnn", overlap=overlap)
    freqDataTest, inputSize = stft_converter(data=x_test, sampling_nperseg=nPerSeg, setting="cnn", overlap=overlap)

    model = cnn2D(inputSize, classes=classes)
    history = model.fit(freqDataTrain, y_train_aug, epochs=epochsCNN, batch_size=batchSize,
                        validation_data=(freqDataTest, y_test),
                        verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    print("Accuracy CNN2D-STFT-Augmented: " + str(accuracy))

if(mlpConcSTFT):
    # N concantentated together
    frequencies = [10, 30, 200]  # Change to whatever frequencies desired
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    concantentatedXTrain = np.array([], dtype=np.float32).reshape(m_train, 0)
    concantentatedXTest = np.array([], dtype=np.float32).reshape(m_test, 0)
    for freq in frequencies:
        freqDataTrain, _ = stft_converter(data= x_train, sampling_nperseg=freq, overlap=overlap)
        freqDataTest, _ = stft_converter(data= x_test, sampling_nperseg=freq, overlap=overlap)
        # print("data shape: ", freqDataTrain.shape)
        concantentatedXTrain = np.concatenate((concantentatedXTrain, freqDataTrain), axis=1)
        concantentatedXTest = np.concatenate((concantentatedXTest, freqDataTest), axis=1)
        # print("data shape Total: ", concantentatedXTrain.shape)
    print("final shape: ", concantentatedXTrain.shape)
    # MLP STFT Combine-2
    clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=model_size, random_state=1, max_iter=epochs)
    clf.fit(np.abs(concantentatedXTrain), y_train)
    y_hat = clf.predict(np.abs(concantentatedXTest))
    print("Accuracy MLP STFT-Concatenated: " + str(accuracy_score(y_test, y_hat)))

if (cnnConcSTFT):
    augmentConc = False
    if(augmentConc):
        x_train = x_train_aug
        y_train = y_train_aug

    # N concantentated together
    frequencies = [10, 100, 180]  # Change to whatever frequencies desired
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    concantentatedXTrain = np.array([], dtype=np.float32).reshape(m_train, 0)
    concantentatedXTest = np.array([], dtype=np.float32).reshape(m_test, 0)
    for freq in frequencies:
        #Use MLP since we're concatenating to 1-Dimensional
        freqDataTrain, _ = stft_converter(data=x_train, sampling_nperseg=freq, overlap=overlap)
        freqDataTest, _ = stft_converter(data=x_test, sampling_nperseg=freq, overlap=overlap)
        concantentatedXTrain = np.concatenate((concantentatedXTrain, freqDataTrain), axis=1)
        concantentatedXTest = np.concatenate((concantentatedXTest, freqDataTest), axis=1)
    print("final shape: ", concantentatedXTrain.shape)

    #CNN2D Concantenated
    inputSize = (concantentatedXTrain.shape[1],1)
    concantentatedXTrain = np.expand_dims(concantentatedXTrain, axis=2)
    concantentatedXTest = np.expand_dims(concantentatedXTest, axis=2)
    print("xtrain shape: ", concantentatedXTrain.shape)
    print("input Size cnn: ", inputSize)
    model = cnn1D(inputSize, classes=classes)
    history = model.fit(concantentatedXTrain, y_train, epochs=epochsCNN, batch_size=batchSize,
                        validation_data=(concantentatedXTest, y_test),
                        verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    print("Accuracy CNN2D-STFT-Concatented-2: " + str(accuracy))
