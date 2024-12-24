"""
This file is to learn and imitate the original process of patient data processing.
(1) From raw data to data input
Source: HERCULES/data_import/importUbisense.ipynb
"""

'''
Initialisation
'''

import pandas as pd
import numpy as np
import os

order_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_list_noweekend = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# df = pd.read_csv('./tech/')
print(os.path.exists(r'data_process1'))
