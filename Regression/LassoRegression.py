import numpy as np


def SoftThreshold(x=np.random.rand(10)-0.5,gamma=0.1):
    y_pos = x-0.5*gamma
    y_neg = x+0.5*gamma

    y = np.multiply(y_pos,y_pos>0) + np.multiply(y_neg,y_neg<0)

    return y

def main():

    SoftThreshold()


if __name__ == '__main__':
    main()