#!/usr/bin/env python3
# ============================================================================
# File:     preprocess_rain_data.py
# Author:   Erik Johannes Husom
# Created:  2019-11-14
# ----------------------------------------------------------------------------
# Description:
#
# ============================================================================
import numpy as np


def load_data():
    filename = 'data/d.csv'
    data = np.loadtxt(open(filename, 'rb'), delimiter=',', skiprows=1)
    print(data)


if __name__ == '__main__': 
    load_data()
