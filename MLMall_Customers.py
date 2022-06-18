import streamlit as st
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

#Distribution of Age
fig = plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df.Age)
plt.title("Distribution of AGE\n=================================================================", fontsize=20, color="black")
plt.xlabel("Age Range", fontsize=15)
plt.ylabel("Density", fontsize=15)

plt.show()
st.pyplot(fig)



plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Annual Income"])
plt.title("Distribution of Annual Income (k$)\n=================================================================", fontsize=20, color="black")
plt.xlabel("Annual Income", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()



plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Spending Score"])
plt.title("Distribution of Spending Score (1-100)\n=================================================================", fontsize=20, color="Black")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()


plt.figure(figsize=(7,5))
sns.set_style('darkgrid')

plt.title("Distribution Gender\n======================================================", fontsize=20, color="Black")
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Count", fontsize=15)
sns.countplot(df.Gender, palette="nipy_spectral_r")
plt.show()


# Age VS Anual Income
plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.scatterplot(data=df, x="Age", y= "Annual Income", hue="Gender", s=60)
plt.title("Age VS Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.show()


# Spending score VS Anual Income

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.scatterplot(data=df, x="Spending Score", y= "Annual Income", hue="Gender", s=60)
plt.title("Spending Score (1-100) VS Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.show()
