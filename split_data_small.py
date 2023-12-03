

import pandas as pd
import os

path = '/users/anair27/data/DiffFolder/DiffFolder/splits/limit256.csv'
df = pd.read_csv(path)
print(len(df))
# 246853 samples in the df
df = df.sample(n = 30000, random_state = 42)
df.to_csv('/users/anair27/data/DiffFolder/DiffFolder/splits/sampledlimit256.csv')

df = df.sample(n = 30, random_state = 42)
df.to_csv('/users/anair27/data/DiffFolder/DiffFolder/splits/sampledlimit256_petite.csv')