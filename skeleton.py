"""
A skeleton for the <USER STATE> component in the ChronoPilot (controller)
architecture.

TLDR:
  Open socket
  Connect to Emotibit
  LOOP
    Get PPG data (25 frames every second)
    Circular buffer (25 * 15s = 375 frames)
    - shift
    - append to end
    Parse
    Send output to socket
"""

__author__ = 'Pieter Van Molle (pieter.vanmolle@ugent.be)'

import atexit
import socket
import time

import brainflow as bf
import neurokit2 as nk
import heartpy as hp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import resample
import joblib

_CONTROLLER_ADDRESS = ('127.0.0.1', 5006)


def _process(data: np.array, freq: float, classifier, verbose=False) -> int:
    """
    Args:
        data: A Numpy array of shape (6, 375), containing biological sensor
              data for the last 15 seconds (sampled at 25Hz).
        freq: The frequency of the data (e.g., 25 Hz).
        classifier: The classifier you want to use for the classification.
        verbose: Whether to show plots or not. False for production.

    Returns:
        cognitive_load: an integer value, representing the load
                        (0 = low, 1 = neutral, 2 = high).
    """

    signal_resampled = data[3]  # PPG_GREEN values
    target_freq = 100  # Hz; upsample to 100 Hz to improve calculation stability

    filtered = hp.filter_signal(signal_resampled, [0.7, 3.5], sample_rate=freq, order=3, filtertype='bandpass')
    signal_resampled = resample(filtered, int(np.ceil(filtered.shape[0] * target_freq / freq)))

    # NOTE: HeartPy implementation: m are the interesting features
    # wd, m = hp.process(signal_resampled, sample_rate=target_freq, high_precision=True, clean_rr=True, bpmmin=0, bpmmax=250)

    # NOTE: NeuroKit2 implementation
    ppg_process, info = nk.ppg_process(signal_resampled, sampling_rate=target_freq)
    ppg_features = nk.ppg_analyze(ppg_process, sampling_rate=target_freq, method="interval-related")
    # # NOTE: We cannot calculate certain features due to the fact that the signal is too short or the sampling
    # #  frequency is to small
    ppg_features.drop(columns=["HRV_SDANN2", "HRV_SDNNI2", "HRV_SDANN5", "HRV_SDNNI5", "HRV_ULF"], inplace=True)
    try:
        ppg_features.drop(columns=['HRV_DFA_alpha2', 'HRV_MFDFA_alpha2_Width', 'HRV_MFDFA_alpha2_Peak',
                                   'HRV_MFDFA_alpha2_Mean', 'HRV_MFDFA_alpha2_Max',
                                   'HRV_MFDFA_alpha2_Delta', 'HRV_MFDFA_alpha2_Asymmetry',
                                   'HRV_MFDFA_alpha2_Fluctuation', 'HRV_MFDFA_alpha2_Increment', ], inplace=True)
    except:
        print("subject does not can calculate HRV_DFA_alpha2 values")

    # print(ppg_features.shape)
    if verbose:
        matplotlib.use('QtAgg')
        # neurokit plotting
        nk.ppg_plot(ppg_process, info)
        # heartpy plotting
        # hp.plotter(wd, m, show=True)
        plt.show()

    # ensure no nan values
    ppg_features.fillna(0, inplace=True)

    # classification

    # the results are the probability distribution over slow, neutral, fast time perception
    res = classifier.predict_proba(ppg_features)

    # selection mechanism (usually the class with the highest probability)
    res_class = int(np.argmax(res))

    return res_class


def run_sensor():
    # The <USER STATE> component is a separate Python process fetching
    # and parsing data from a biological sensor (such as the Emotibit).
    # Parsed results, i.e. the cognitive load, are communicated to the
    # <CONTROLLER> using UDP datagrams.

    # Cognitive load has three (3) discrete values, being
    # - low,
    # - neutral,
    # - high.

    # Set up a UDP socket, used for sending text to the controller
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Messages must follow a fixed pattern, having a preamble (between $-signs),
    # containing a source name and a incrementing message number.
    # The preamble is `$EmotibitState <message number>$`.
    message_number = 0

    # Load the classifier model
    # TODO adjust path
    clf = joblib.load("helicopter_3_classes.pkl")

    # The actual body of the message is
    # `state cognitive_load load=[low|neutral|high]`, directly following the
    # preamble.
    # NOTE: while almost final, this syntax is subject to change, as it is
    # currently being drafted by Uni Luxembourg

    # Use the Brainflow library to skip the Emotibit Oscilloscope and
    # communicate directly with the Emotibit board.
    emoti_idx = bf.BoardIds.EMOTIBIT_BOARD
    params = bf.BrainFlowInputParams()
    board = bf.BoardShim(emoti_idx, params)

    def _cleanup():
        try:
            board.stop_stream()
            board.release_all_sessions()
        except bf.exit_codes.BrainFlowError:
            pass

    atexit.register(_cleanup)

    # Actual board connection
    connected = False
    attempts = 3
    while not connected and attempts > 0:
        try:
            board.prepare_session()
            connected = True
        except bf.exit_codes.BrainFlowError:
            attempts -= 1

    if attempts == 0:
        print('Error connecting to Emotibit')
        exit(1)

    # Using the Brainflow library, the PPG data can be fetched periodically.
    # At a sampling rate of 25Hz, every second, we want to pull 25 frames,
    # and append these to a circular buffer. After an initialization phase,
    # this buffer will contain a windowed view of the data of the last 15
    # seconds.
    # TODO: we need to make the time window larger:
    #  - 15 seconds is too short for nk
    #  - 30 seconds is too short for nk
    #  - 45 seconds could work for nk
    #  However, larger windows allow for more stable HRV calculations
    time_window = 45  # seconds
    freq = 25  # Hz  # NOTE: this is the sampling rate of the Emotibit
    buffer = np.zeros((6, time_window * 25), dtype=np.float64)

    board.start_stream()

    # Initial "warm up" period
    time.sleep(2)

    seconds_passed = 0

    # NOTE: for debugging purposes, we use random data
    # sample_data = pd.read_csv("/Volumes/Data/chronopilot/test.csv")
    # sample_data.dropna(inplace=True)
    # pointer = 0
    # pointer1 = 25
    # freq = sample_data.shape[0] / (sample_data['LocalTimestamp'].iloc[-1] - sample_data['LocalTimestamp'].iloc[0])
    # print(
    #     f"sample frequence: {sample_data.shape[0] / (sample_data['LocalTimestamp'].iloc[-1] - sample_data['LocalTimestamp'].iloc[0])}")

    while True:
        time.sleep(1)
        seconds_passed += 1

        data = board.get_current_board_data(
            25, preset=bf.BrainFlowPresets.AUXILIARY_PRESET)

        # NOTE this is only for debugging to have a data stream
        # data = np.random.rand(6, 25)
        # data[3] = sample_data["PG"].iloc[pointer:pointer1].to_numpy()
        # pointer = (pointer + 25)
        # pointer1 = (pointer1 + 25)
        # if pointer1 > sample_data.shape[0]:
        #     pointer1 = 25
        #     pointer = 0

        # The fetched data will be a Numpy array of shape (6, x), where x is
        # (at most) the requested number of frames. In the data, the rows
        # represent:
        #   0: package number,
        #   1: PPG_INFRARED,
        #   2: PPG_RED,
        #   3: PPG_GREEN,
        #   4: timestamp (host time, different from Emotibit time),
        #   5: marker.
        assert data.shape == (6, 25)

        # Append to circular buffer
        buffer[:, :-25] = buffer[:, 25:]
        buffer[:, -25:] = data

        # Process the results
        # (starting after the buffer is filled a first time,
        # i.e. after 15 seconds)
        # NOTE: currently, we assume processing of PPG data is near-instant
        if seconds_passed >= time_window:
            res = _process(buffer, freq, clf, verbose=False)
            load = ('low', 'neutral', 'high')[res]
            print(f"Cognitive load: {load}")
            payload = (
                f'$EmotibitState {message_number}$'
                f'state cognitive_load load={load}')

            # Send result to controller
            sock.sendto(payload.encode('utf-8'), _CONTROLLER_ADDRESS)
            message_number += 1


if __name__ == '__main__':
    run_sensor()
