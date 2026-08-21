import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/encoded.csv")

df.hist(figsize=(20,20))

plt.tight_layout()

plt.show()