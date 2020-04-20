import numpy as np

def crop(frame):
    # Crop to 84x84
    return frame[:-12, 6:-6]


def rgb2grayscale(frame):
    # change to grayscale
    return np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])