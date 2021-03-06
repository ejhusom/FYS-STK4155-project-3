#!/usr/bin/env python3
# ============================================================================
# File:     featprint
# Author:   Erik Johannes Husom
# Created:  2019-12-05
# ----------------------------------------------------------------------------
# Description:
# Save feature importance as numpy arrays.
# ============================================================================
import numpy as np 

feature_importance = {'tree': [0.04119991, 0.05167874, 0.12805375, 0.00454587,
    0.1973286,  0.00739907, 0.00438991, 0.00442739, 0.00277796, 0.08245902,
    0.09266789, 0.13685822, 0.0550103,  0.08147789, 0.08418555, 0.02553992,],
    'bagging': [0.05641168, 0.04191471, 0.08184195, 0.00846297, 0.1960321,
        0.00951002, 0.00454187, 0.00687586, 0.00367635, 0.09022062, 0.08253264,
        0.12991216, 0.06743362, 0.07035134, 0.0808825,  0.06939961], 'random':
    [0.06808629, 0.05378452, 0.07542956, 0.03025101, 0.06268223, 0.03397167,
        0.02389375, 0.04879707, 0.03624232, 0.10035244, 0.09986861, 0.10539286,
        0.05962506, 0.07273982, 0.05307037, 0.07581242]}

tree = np.array(feature_importance['tree'])
bagging = np.array(feature_importance['bagging'])
randomforest = np.array(feature_importance['random'])

np.save('featimp-decisiontree-case1.npy', tree)
np.save('featimp-bagging-case1.npy', bagging)
np.save('featimp-randomforest-case1.npy', randomforest)
a = np.load('featimp-decisiontree-case1.npy')
b = np.load('featimp-bagging-case1.npy')
c = np.load('featimp-randomforest-case1.npy')

tree_case2 = [0.07068566, 0.03171934, 0.02256834, 0.0058181,  0.1833569,  0.00891043,
 0.00281869, 0.00478877, 0.00357627, 0.091519,   0.17538176, 0.10870657,
 0.0761631,  0.07178151, 0.02735275, 0.11485281]
bagging_case2 = [0.06233289, 0.03632528, 0.04900637, 0.01413281, 0.18380748, 0.00859397,
 0.00407993, 0.00560589, 0.00302398, 0.10674585, 0.13095476, 0.1355339,
 0.05684061, 0.0919102,  0.03198606, 0.07912002]
random_case2 = [0.06956644, 0.05235452, 0.07125713, 0.0212575,  0.06263269, 0.02973148,
 0.03313004, 0.04892518, 0.03153263, 0.09950679, 0.10636449, 0.11232948,
 0.06323731, 0.07206387, 0.05282183, 0.0732886 ]

np.save('featimp-decisiontree-case2.npy', tree_case2)
np.save('featimp-bagging-case2.npy', bagging_case2)
np.save('featimp-randomforest-case2.npy', random_case2)


print(a)
print('---------')
print(b)
print('---------')
print(c)



