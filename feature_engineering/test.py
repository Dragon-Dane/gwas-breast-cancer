import numpy as np
import pandas as pd

a = np.random.rand(3, 4)
df = pd.DataFrame(a)
print(df)

col3= df[3]
print(col3)

df['class'] = col3
print(df)