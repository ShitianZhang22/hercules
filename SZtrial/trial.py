import numpy as np
import pandas as pd

a = '12/1/2022, 12:02:20 AM'
a = pd.to_datetime(a).strftime('%Y-%m-%d %H:%M:%S')
print(a)
print(type(a))
