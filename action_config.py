import numpy as np

VALID_ACTIONS = np.arange(0,10)


def idx2act(num):
    """
    Convert action index to action input for racing-car-v0
    Discretize the action space in this function
    """
    steer = 0.0
    gas = 0.1
    brake = 0.0
    if (num < 7):
        steer = (num - 3) / 3
    if (num == 7):
        gas = 0.5
    if (num == 8):
        gas = 1
    if (num == 9):
        gas = 0
        brake = 0.8

    return [steer, gas, brake]