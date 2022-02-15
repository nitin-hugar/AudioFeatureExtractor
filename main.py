
"""
Script to extract multiple features from audio and save them to a file

Author: Nitin Hugar
"""

#===============================================================================
# Import dependancies
#===============================================================================

import madmom
import librosa
import numpy as np
import soundfile as sf
import scipy.stats
from madmom.audio.chroma import DeepChromaProcessor
from madmom.processors import SequentialProcessor
from scipy import signal
from scipy.signal import butter, filtfilt, medfilt
from scipy.fft import fft
import msaf
import math
import time

#===============================================================================
# Function Definitions 
#===============================================================================


# Function imports audio from a given audio file path
def import_audio(audioFilePath):
    data, samplingrate = sf.read(audioFilePath)
    if data.shape[1] == 2:
        data = data[:, 0]
    return data, samplingrate

# Function blocks audio into chunks
def block_audio(x, blockSize, hopSize, fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t

# Function computes a fourier transform of an audio block and returns the magnitude spectrum
def fourier(x):
    # Get Symmetric fft
    w = signal.windows.hann(np.size(x))
    windowed = x * w
    w1 = int((x.size + 1) // 2)
    w2 = int(x.size / 2)
    fftans = np.zeros(x.size)

    # Centre to make even function
    fftans[0:w1] = windowed[w2:]
    fftans[w2:] = windowed[0:w1]
    X = fft(fftans)
    magX = abs(X[0:int(x.size // 2 + 1)])
    return magX

# Function extracts feature "Spectral crest" of audio block
def extract_spectral_crest(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1] / 2 + 1)))
    spc = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        magX[block] = fourier(xb[block])
        summa = np.sum(magX[block], axis=0)
        if not summa:
            summa = 1
        spc[block] = np.max(magX[block]) / summa
    return spc

# Function extracts feature "Spectral flux" of audio block
def extract_spectral_flux(xb):
    magX = np.zeros((xb.shape[0], int(xb.shape[1] / 2 + 1)))
    specflux = np.zeros((xb.shape[0]))
    magX[0] = fourier(xb[0])
    for block in np.arange(1, xb.shape[0]):
        magX[block] = fourier(xb[block])
        den = magX[block].shape[0]
        specflux[block] = np.sqrt(np.sum(np.square(magX[block] - magX[block - 1]))) / den
    return specflux

# Function returns onset times in an audio file 
def detect_onsets(audioFilePath):
    proc = madmom.features.onsets.OnsetPeakPickingProcessor(fps=100)
    act = madmom.features.onsets.RNNOnsetProcessor()(audioFilePath)
    onsets = proc(act)
    return onsets

# Function returns beat times in an audio file
def detect_beats(audioFilePath):
    b = madmom.features.beats.RNNBeatProcessor()(audioFilePath)
    beats = madmom.features.beats.BeatTrackingProcessor(fps=100)(b)
    return beats

# Function returns downbeats in an audio file
def detect_downbeats(audioFilePath):
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=4, min_bpm=60, max_bpm=160, correct=True, fps=100)  # check
    act = madmom.features.downbeats.RNNDownBeatProcessor()(audioFilePath)
    downbeats = proc(act)
    return downbeats

# Function returns tempo of an audio file
def detect_tempo(audioFilePath):
    x, fs = import_audio(audioFilePath)
    tempo, beats = librosa.beat.beat_track(x, sr=fs, units='time') # 'units : 'time', 'frames', 'samples'
    return tempo 
    
# Function returns key of an audio file
def detect_key(audioFilePath):
    key_probabilities = madmom.features.key.CNNKeyRecognitionProcessor()
    key_prediction = madmom.features.key.key_prediction_to_label(key_probabilities(audioFilePath))
    return key_prediction

# DeepChromaChordRecognition
def detect_chords_deep_chroma(audioFilePath):
    dcp = DeepChromaProcessor()
    decode = madmom.features.chords.DeepChromaChordRecognitionProcessor()
    chordrec = SequentialProcessor([dcp, decode])
    chords = chordrec(audioFilePath)
    return chords

# CRF chord RecognitionProcessor
def detect_chords_CRF(audioFilePath):
    featproc = madmom.features.chords.CNNChordFeatureProcessor()
    decode = madmom.features.chords.CRFChordRecognitionProcessor()
    chordrec = SequentialProcessor([featproc, decode])
    chords = chordrec(audioFilePath)
    return chords

# Function extracts rms feature from audio blocks 
def extract_rms(data, blockSize, hopSize, fs):
    xb, t = block_audio(data, blockSize, hopSize, fs)
    rms = np.zeros(xb.shape[0])
    for block in range(xb.shape[0]):
        if not np.all((xb[block] == 0)): # Check for zero errors
            rms[block] = np.sqrt(np.mean(np.square(xb[block])))
            rms[block] = 20 * np.log10(rms[block])
            if (rms[block] < -90):
                rms[block] = -90
        else: 
            rms[block] = -90
    return rms

# creates a voicing mask which is 1 when the audio level is above or equal to threhold and 0 when below
def detect_silence(data, blockSize, hopSize, fs, thresholdDb):
    rmsDb = extract_rms(data, blockSize, hopSize, fs)
    mask = np.zeros(rmsDb.shape[0])
    for i in range(rmsDb.shape[0]):
        if rmsDb[i] >= thresholdDb:
            mask[i] = 1
        else:
            mask[i] = 0
    return mask


# Function extracts section boundaries and labels of an audio file
def extract_sections(audioFilePath):
    boundaries, labels = msaf.process(audioFilePath, sonify_bounds=False,
                                        boundaries_id='foote', labels_id="cnmf")
    return boundaries, labels

# Helper function converts frequency to midi
def freq2midi(freq):
    midi = 69 + 12 * np.log2(freq / 440)
    return np.asarray(midi, dtype=np.int32)

# Helper function comverts midi values to notes
def midi_to_notes(midi, fs, hopSize, smooth, minduration):
    # smooth midi pitch sequence first
    if smooth > 0:
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hopSize))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi

    notes = []
    p_prev = 0
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hopSize / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hopSize / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hopSize / float(fs)
        onset_sec = onset * hopSize / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes

# Modify to include only notes from the obtained key and also use tempo determined from the song
def pitch_to_midi(data, fs, hopSize):

    # low pass filter_audio
    filter_order = 4
    frequency = 1000.
    lowpass_f = frequency / (fs / 2.)
    b, a = butter(filter_order, lowpass_f, 'low')
    filtered_melody = filtfilt(b, a, data)
    print("Getting fundamental contour")
    # get fundamental contour
    f0_lowpass, voiced_flag_low, voiced_probs_low = librosa.pyin(filtered_melody, sr=fs,
                                                                    fmin=librosa.note_to_hz('C2'),
                                                                    fmax=librosa.note_to_hz('C7'))
    print("Getting midi notes")
    midiNotes = freq2midi(f0_lowpass)
    print("converting midi to note values")
    notes = midi_to_notes(midiNotes, fs, hopSize=hopSize, smooth=0.40, minduration=0.1)
    return notes

#===============================================================================
# Main function
#===============================================================================

def main():
   
    # Comment out any feature that you do not want to extract

    audioFilePath = input("Enter audio filepath: ")

    start = time.time()
    print("Importing file...")
    data, fs = import_audio(audioFilePath)

    print("Extracting onsets...")
    onsets = detect_onsets(audioFilePath)

    print("Extracting Beats...")
    beats = detect_beats(audioFilePath)

    print("Extracting Downbeats...")
    downbeats = detect_downbeats(audioFilePath)

    print("Extracting Tempo...")
    tempo = detect_tempo(audioFilePath)

    print("Extracting Key...")
    key = detect_key(audioFilePath)

    print("Extracting Chords...")
    chords = detect_chords_deep_chroma(audioFilePath)

    print("Extracting RMS...")
    rms = extract_rms(data, 1024, 512, fs)

    print("Extracting Silence...")
    silence = detect_silence(data, 1024, 512, fs, thresholdDb=-40)

    print("Extracting Boundaries...")
    boundaries, _ = extract_sections(audioFilePath)
    notes = pitch_to_midi(data, fs, hopSize=512)
    print(notes)
    end = time.time()
    print(end-start)
    print("Done!")
    
    feature_dict = {}
    feature_dict['onsets'] = onsets
    feature_dict['beats'] = beats
    feature_dict['downbeats'] = downbeats
    feature_dict['tempo'] = tempo
    feature_dict['key'] = key
    feature_dict['chords'] = chords
    feature_dict['rms'] = rms
    feature_dict['silence'] = silence
    feature_dict['boundaries'] = boundaries
    
    with open('feature_dict.csv', 'w') as f:
        for key in feature_dict.keys():
            f.write("%s,%s\n" % (key, feature_dict[key]))


if __name__ == '__main__': 
    main()




