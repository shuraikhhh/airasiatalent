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


# Distribution of Anual Income
fig1 = plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Annual Income"])
plt.title("Distribution of Annual Income (k$)\n=================================================================", fontsize=20, color="black")
plt.xlabel("Annual Income", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()
st.pyplot(fig1)



# Distribution of Spending score 
fig2 = plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.distplot(df["Spending Score"])
plt.title("Distribution of Spending Score (1-100)\n=================================================================", fontsize=20, color="Black")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Density", fontsize=15)
plt.show()
st.pyplot(fig2)


# Distribution of Gender
fig3 = plt.figure(figsize=(7,5))
sns.set_style('darkgrid')

plt.title("Distribution Gender\n======================================================", fontsize=20, color="Black")
plt.xlabel("Gender", fontsize=15)
plt.ylabel("Count", fontsize=15)
sns.countplot(df.Gender, palette="nipy_spectral_r")
plt.show()
st.pyplot(fig3)


# Age VS Anual Income

fig4 = plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.scatterplot(data=df, x="Age", y= "Annual Income", hue="Gender", s=60)
plt.title("Age VS Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Age", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.show()
st.pyplot(fig4)


# Spending score VS Anual Income

fig5 = plt.figure(figsize=(10,6))
sns.set_style('darkgrid')

sns.scatterplot(data=df, x="Spending Score", y= "Annual Income", hue="Gender", s=60)
plt.title("Spending Score (1-100) VS Annual Income (k$)\n=================================================================", fontsize=20, color="green")
plt.xlabel("Spending Score (1-100)", fontsize=15)
plt.ylabel("Annual Income (k$)", fontsize=15)
plt.show()
st.pyplot(fig5)


#Scaling before applying model
df_scaled = df[["Age","Annual Income","Spending Score"]]

# Class instance
scaler = StandardScaler()      #WHY WE USE STANDARD SCALER

# Fit_transform
df_scaled_fit = scaler.fit_transform(df_scaled)


df_scaled_fit = pd.DataFrame(df_scaled_fit)
df_scaled_fit.columns = ["Age","Annual Income","Spending Score"]
var_list = df_scaled_fit[["Annual Income","Spending Score"]]

# Modelling k means
kmeans = KMeans(n_clusters=5, max_iter=50)
kmeans.fit(var_list)
kmeans.labels_

#appending labels column to df
df["Label"] = kmeans.labels_ #appending label cloumn to df
df.head()


#plotting data
fig6 = plt.figure(figsize=(10,6))
plt.title("Plotting the data\n============================================================", fontsize=20,color='black')
plt.scatter(df['Annual Income'],df['Spending Score'],color='blue')
st.pyplot(fig6)

fig7 = plt.figure(figsize=(10,6))

plt.title("Ploting the data into 5 clusters\n=================================================================", fontsize=20, color="black")
sns.scatterplot(data=df, x="Annual Income", y="Spending Score", hue="Label", s=60, palette='Set2')
plt.show()
st.pyplot(fig7)
