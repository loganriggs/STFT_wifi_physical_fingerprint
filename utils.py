import numpy as np
import scipy.signal
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#Dr. Ball's Augment Code
def Augment_complex(X, Y, num_augments, augment_method=1, aug_std_dev=5.0e-5):
    Xaugment = np.zeros((X.shape[0] * (num_augments + 1), X.shape[1]), dtype=np.complex64)
    Yaugment = np.zeros(Y.shape[0] * (num_augments + 1), dtype=int)

    idx = 0
    tempX = np.zeros((1, X.shape[1]), dtype=np.complex64)
    for k in np.arange(X.shape[0]):
        tempX[0, :] = X[k, :]
        for aug in np.arange(num_augments + 1):
            if aug > 0:
                tempX[0, :] = X[k, :]

                if augment_method == 1: #Random signal
                    tempX += aug_std_dev * (np.random.randn(1, tempX.shape[1]) +
                                            np.random.randn(1, tempX.shape[1]) * 1j)
                else: #Random Vector
                    tempX = tempX * np.exp(np.random.rand(1, 1) * 2.0 * np.pi * 1j)
            Xaugment[idx, :] = tempX[0, :]
            Yaugment[idx] = Y[k]
            idx = idx + 1

    X = Xaugment
    Y = Yaugment.astype(int)

    return X, Y

def stft_converter(data,  sampling_nperseg, frequency=1, setting="mlp", overlap = 0):
    #Get input size by running STFT once
    f, t, Zxx = scipy.signal.stft(data[0], fs=frequency, nperseg=sampling_nperseg, noverlap=overlap, return_onesided=False)

    if(setting == "mlp"):
        #input size = Flatten version
        inputSize = Zxx.flatten().shape
        # print("Input Size MLP: ", inputSize)
        m = data.shape[0]
        mTuple = (m,)
        freqData = np.zeros(mTuple + inputSize, dtype='float32')
        # Create actual dataset
        for i in range(0, m):
            f, t, Zxx = scipy.signal.stft(data[i], fs=frequency, nperseg=sampling_nperseg, noverlap=overlap, return_onesided=False)
            freqData[i, :] = np.abs(Zxx.flatten()).astype("float32")


    elif(setting == "cnn"):
        inputSize = Zxx.shape
        # print("Input Size CNN: ", inputSize)
        m = data.shape[0]
        mTuple = (m,)
        freqData = np.zeros(mTuple + inputSize, dtype='float32')

        # Create actual STFT dataset
        for i in range(0, m):
            f, t, Zxx = scipy.signal.stft(data[i], fs=frequency, nperseg=sampling_nperseg, noverlap=overlap, return_onesided=False)
            freqData[i, :, :] = np.abs(Zxx).astype("float32")

        #Expand Dimension for CNN2D shape compliance
        freqData = np.expand_dims(freqData, axis=3)
    return (freqData, inputSize)

def cnn2D(inputSize, opt = 'adam', loss = SparseCategoricalCrossentropy(from_logits=True), classes = 10):
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
    return model

def cnn1D(inputSize, opt = 'adam', loss = SparseCategoricalCrossentropy(from_logits=True), classes = 10):
    model = models.Sequential()
    # 81%
    model.add(layers.Conv1D(32, (5), activation='relu', padding="same", input_shape=inputSize))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.MaxPooling1D((2)))
    model.add(layers.Conv1D(64, (3), padding="same", activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.MaxPooling1D((2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.5))
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    return model

