import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('https://raw.githubusercontent.com/shuraikhhh/airasiatalent/main/Mall_Customers.csv')

df.rename(columns={"Genre":"Gender"}, inplace=True)
df.rename(columns={"Annual Income (k$)":"Annual Income"}, inplace=True)
df.rename(columns={"Spending Score (1-100)":"Spending Score"}, inplace=True)

df
