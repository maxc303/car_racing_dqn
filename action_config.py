import numpy as np

VALID_ACTIONS = np.arange(0,10)


def idx2act(num):
    '''
    convert action index to action input for racing-car-v0

    '''
    # ###Ok Action config
    # steer = 0.0
    # gas = 0.0
    # brake = 0.0
    # if (num < 7):
    #     steer = (num - 3) / 3
    # if (num == 7):
    #     gas = 0.5
    # if (num == 8):
    #     gas = 1
    # if (num == 9):
    #     gas = 0
    #     brake = 0.8

    ###AVG 800
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