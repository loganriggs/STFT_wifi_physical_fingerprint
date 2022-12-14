import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google_drive_downloader import GoogleDriveDownloader as gdd


#Settings: ant1, ant2, csi
# "ant1" = antenna 1
# "ant2" = antenna 1 & 2
# "csi" =  antenna 1 & 2 & CSI
setting = "ant1"


path_to_file = "./data/"
file_id = 'classificationWiFiDeviceFP.hdf5'

try:
    hf = h5py.File(path_to_file + file_id, 'r')
    print("Already downloaded file. Skipping...")
except OSError:
    print("File not found. Downloading file...")
    gdd.download_file_from_google_drive(file_id='19BwVlX3ABtV1CzOQko2lUj9bU2zEpNOA',
                                        dest_path= path_to_file + file_id,
                                        unzip=True)

    # download_file_from_google_drive("https://drive.google.com/file/d/19BwVlX3ABtV1CzOQko2lUj9bU2zEpNOA/view?usp=sharing", path_to_file + file_id)
    hf = h5py.File(path_to_file + file_id, 'r')


ant1_raw_frames = hf.get('ant1Train')[()]
transmitter_labels = hf.get('dev_labels')[()]
y = transmitter_labels

if(setting == "ant1"):
    x = ant1_raw_frames
elif(setting == "ant2"):
    ant2_raw_frames = hf.get('ant2Train')[()]
    x = np.concatenate((ant1_raw_frames, ant2_raw_frames), axis=1)
elif(setting == "csi"):
    ant2_raw_frames = hf.get('ant2Train')[()]
    csi_frames = hf.get('csiTrain')[:, 0:128]
    #Grab the first 128 samples from csi
    csi_preamble = csi_frames[:, 0:128]
    x = np.concatenate((ant1_raw_frames, ant2_raw_frames, csi_preamble), axis=1)


#Split into training and test and save data
#Note: Random state is selected so the same train-test split is used

x_train, x_test, y_train,  y_test = train_test_split(x, y, stratify=y, test_size=0.33, random_state=42)
np.save("classification_x_train", x_train)
np.save("classification_x_test", x_test)
np.save("classification_y_train", y_train)
np.save("classification_y_test", y_test)

# Other Values
# rssi_train = hf.get('powTrain')[()]
# location_labels = hf.get('loc_labels')[()]
# receiver_labels = hf.get('rx_labels')[()]

# Plot abs of 1 signal
# plt.rcParams.update({'font.size': 16})
# plt.plot(np.abs(ant1_raw_frames[1]))
# plt.title("802.11 a/g Wireless Signal")
# plt.xlabel("Sample")
# plt.ylabel("Magnitude")
# plt.show()
