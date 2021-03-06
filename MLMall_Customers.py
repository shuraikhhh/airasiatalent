import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.header("Mall_Customers K-means clustering")
st.write("By Shuraikh 'Ezzuddin")

df = pd.read_csv('https://raw.githubusercontent.com/shuraikhhh/airasiatalent/main/Mall_Customers.csv')

df.rename(columns={"Genre":"Gender"}, inplace=True)
df.rename(columns={"Annual Income (k$)":"Annual Income"}, inplace=True)
df.rename(columns={"Spending Score (1-100)":"Spending Score"}, inplace=True)

#st.write("Mall Customer Dataset")
#st.dataframe(df)
option = st.sidebar.selectbox(
    'Select:',
     ['Dataset','K-means clustering', 'Distribution of Data', 'Bi-variate Analysis'])

if  option=='Dataset':
    st.write("Mall Customer Dataset")
    st.table(df)

elif  option=='Distribution of Data':
    #Distribution of Age
    fig = plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')

    sns.distplot(df.Age)
    plt.title("Distribution of AGE\n", fontsize=18, color="black")
    plt.xlabel("Age Range", fontsize=15)
    plt.ylabel("Density", fontsize=15)

    plt.show()
    st.pyplot(fig)
    st.markdown('##')


    # Distribution of Annual Income
    fig1 = plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')

    sns.distplot(df["Annual Income"])
    plt.title("Distribution of Annual Income (k$)\n", fontsize=18, color="black")
    plt.xlabel("Annual Income", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.show()
    st.pyplot(fig1)
    st.markdown('##')



    # Distribution of Spending score 
    fig2 = plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')

    sns.distplot(df["Spending Score"])
    plt.title("Distribution of Spending Score (1-100)\n", fontsize=18, color="Black")
    plt.xlabel("Spending Score (1-100)", fontsize=15)
    plt.ylabel("Density", fontsize=15)
    plt.show()
    st.pyplot(fig2)
    st.markdown('##')


    # Distribution of Gender
    fig3 = plt.figure(figsize=(7,5))
    sns.set_style('darkgrid')

    plt.title("Distribution of Gender\n", fontsize=18, color="Black")
    plt.xlabel("Gender", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    sns.countplot(df.Gender, palette="nipy_spectral_r")
    plt.show()
    st.pyplot(fig3)
    st.markdown('##')

elif  option=='Bi-variate Analysis':
    # Age VS Anual Income

    fig4 = plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')

    sns.scatterplot(data=df, x="Age", y= "Annual Income", hue="Gender", s=60)
    plt.title("Age VS Annual Income (k$)\n", fontsize=18, color="Black")
    plt.xlabel("Age", fontsize=15)
    plt.ylabel("Annual Income (k$)", fontsize=15)
    plt.show()
    st.pyplot(fig4)
    st.markdown('##')


    # Spending score VS Anual Income

    fig5 = plt.figure(figsize=(10,6))
    sns.set_style('darkgrid')

    sns.scatterplot(data=df, x="Spending Score", y= "Annual Income", hue="Gender", s=60)
    plt.title("Spending Score (1-100) VS Annual Income (k$)\n", fontsize=18, color="Black")
    plt.xlabel("Spending Score (1-100)", fontsize=15)
    plt.ylabel("Annual Income (k$)", fontsize=15)
    plt.show()
    st.pyplot(fig5)
    st.markdown('##')

elif  option=='K-means clustering':    

    #Scaling before applying model
    df_scaled = df[["Age","Annual Income","Spending Score"]]

    # Class instance
    scaler = StandardScaler()      #WHY WE USE STANDARD SCALER

    # Fit_transform
    df_scaled_fit = scaler.fit_transform(df_scaled)


    df_scaled_fit = pd.DataFrame(df_scaled_fit)
    df_scaled_fit.columns = ["Age","Annual Income","Spending Score"]
    var_list = df_scaled_fit[["Annual Income","Spending Score"]]

#finding the optimal k value
    ssd = []

    for num_clusters in range(1,11):
        kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
        kmeans.fit(var_list)

        ssd.append(kmeans.inertia_)


    figk = plt.figure(figsize=(12,6))
    st.subheader("Finding the optimal k-value using elbow method")
    plt.plot(range(1,11), ssd, linewidth=2, color="purple", marker ="8")
    plt.title("Elbow Method\n", fontsize=18, color="black")
    plt.xlabel("K Value")
    plt.xticks(np.arange(1,11,1))
    plt.ylabel("SSD")

    plt.show()
    st.pyplot(figk)
    st.markdown('#')
    
    st.write("Based on the graph, we conclude the optimal value of k is 5")

    # Modelling k means
    kmeans = KMeans(n_clusters=5, max_iter=50)
    kmeans.fit(var_list)
    #kmeans.labels_

    #appending labels column to df
    df["Label"] = kmeans.labels_ #appending label cloumn to df
    #df.head()
    st.markdown('#')

    st.subheader("Modelling K-means")
    #plotting data
    fig6 = plt.figure(figsize=(10,6))
    plt.title("Plotting the data\n", fontsize=18,color='black')
    plt.scatter(df['Annual Income'],df['Spending Score'],color='blue')
    st.pyplot(fig6)
    st.markdown('#')

    
    fig7 = plt.figure(figsize=(10,6))
  
    st.subheader("Clustering k=5")
    #clustering
    plt.title("Ploting the data into 5 clusters\n", fontsize=18, color="black")
    sns.scatterplot(data=df, x="Annual Income", y="Spending Score", hue="Label", s=60, palette='Set2')
    plt.show()
    st.pyplot(fig7)
    st.markdown('#')

    df1 = df
    
    st.subheader("Pairplot")
    fig9 =sns.pairplot(df1, hue='Label', aspect=1.5, palette='Paired')
    plt.show()
    st.pyplot(fig9)
    st.markdown('#')

    st.subheader("Relation to annual income and scoring history")
    #relation to annual income and scoring history
    fig8 = plt.figure(figsize=(20,8))
    ax = fig8.add_subplot(121)
    sns.boxplot(x='Label', y='Annual Income', data=df, ax=ax, palette='Set2')
    ax.set_title('Labels According to Annual Income')

    ax = fig8.add_subplot(122)
    sns.boxplot(x='Label', y='Spending Score', data=df, ax=ax,palette='Set2')
    ax.set_title('Labels According to Scoring History')

    plt.show()
    st.pyplot(fig8)
    st.markdown('#')
