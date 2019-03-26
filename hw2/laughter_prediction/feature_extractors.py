import os
import tempfile

import numpy as np
import pandas as pd
# import laughter_classification.psf_features as psf_features
import librosa


EPS = 1e-7


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file
        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""
    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class MFCCFeatureExtractor(FeatureExtractor):
    def __init__(self, steps_per_second):
        super(MFCCFeatureExtractor, self).__init__()
        self.steps_per_second = steps_per_second

    def extract_features(self, wav_path):
        track, sr = librosa.load(wav_path)
        step = int(sr / self.steps_per_second + EPS)
        mel_spectrogram = librosa.feature.melspectrogram(y=track, sr=sr, n_mels=40, hop_length=step, n_fft=2 * step)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, n_mfcc=13)
        return mel_spectrogram, mfcc
