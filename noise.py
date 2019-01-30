import argparse
import os
import numpy as np
import librosa
import soundfile as sf


def add_noise(data, noise, coefficient):
    long_noise = np.zeros_like(data)
    pos = 0
    n = len(long_noise)
    m = len(noise)
    while pos < n:
        step = min(m, n - pos)
        if len(long_noise.shape) == 1:
            long_noise[pos:pos + step] = noise[:step]
        else:
            long_noise[pos:pos + step] = np.expand_dims(noise[:step], 1)
        pos += step
    return data + long_noise * coefficient


def load_file(name):
    if name[-4:] == '.waw':
        return librosa.load(name)
    else:
        return sf.read(name)[::-1]


def save_file(name, data, sr):
    if name[-4:] == '.waw':
        librosa.output.write_wav(name, data, sr)[::-1]
    else:
        sf.write(name, data, sr, format='flac')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('source_file')
    argument_parser.add_argument('target_file')
    argument_parser.add_argument('-n', default=None)
    argument_parser.add_argument('-c', default=0.1, type=float)
    arguments = argument_parser.parse_args()

    sr, sound = load_file(arguments.source_file)
    if arguments.n is None:
        directory = os.path.join('samples', 'beeps' if np.random.binomial(1, .5) == 1 else 'music')
        file = os.path.join(directory, np.random.choice(os.listdir(directory)))
        _, noise= load_file(file)
    else:
        _, noise = load_file(arguments.n)
    noised_sound = add_noise(sound, noise, arguments.c)
    save_file(arguments.target_file, noised_sound, sr)


if __name__ == '__main__':
    main()
