import numpy as np
import pandas as pd

a = '10:22:22'
a = pd.to_datetime(a)
print(a)
print(a.time() > pd.to_datetime('9:22:22').time())
