This is the corresponding code for Smith et. al's "Effect of the Short Time Fourier Transform on the Classification of Complex-Valued Mobile Signals"

To get started, first install the dependencies:

    '''
     pip3 install -r requirements.txt
     '''

To load a specific dataset, change the "setting parameter in "load_h5py_dataset.py" and run this file.
(note: the first run will take longer in order to download the dataset. Ensure you have an internet connection)

    '''
    #Settings: ant1, ant2, csi
    # "ant1" = antenna 1
    # "ant2" = antenna 1 & 2
    # "csi" =  antenna 1 & 2 & CSI
    setting = "ant1"
    '''

Our experiments our located in "mlp_settings.py" and "stft_window_length_parameter_sweep.py". Relevant parameters can be changed at the top of each file.