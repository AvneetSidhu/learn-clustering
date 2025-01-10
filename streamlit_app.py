import streamlit as st
import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("K-Means Clustering Demo - Mall Segmentation")

data = pd.read_csv('Mall_Customers.csv')

st.write("""This is the original data that we are going to perform k-means clustering on. 
            This data is the spending habits of customers along with their age, gender and annual income.
            We are going to perform transformations on the data to make sure it is ready to be processed by the k-means model to hopefully gain some business insights.""")
st.dataframe(data, use_container_width=True)

data = data.drop('CustomerID', axis=1)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

st.write("The first thing we can do to prepare the data is to drop the values of the table that we do not need. In this case it is the CustomerID column.")
st.dataframe(data, use_container_width=True)

scaler = StandardScaler()

features = ["Gender","Age", "Annual Income (k$)", "Spending Score (1-100)"]

scaled = scaler.fit_transform(data[features])

scaled_data = pd.DataFrame(scaled, columns = features)

st.write("""Next we will have to scale the data such that the individual columns/features will have a mean of 0 and a variance of 1.
            This is because we want our features to carry the same weight when it comes to the distance calculations between datapoints. 
            We want the gender, age, and income to all be equally weighed.""")
st.dataframe(scaled_data,  use_container_width=True)

st.write("""After our data is prepared, we need to figure out how many clusters we want to use for clustering.
            We can do this by using the elbow method, where we plot the outputs of the """)

#elbow curve
k = range(1, 10)
inert = []
for element in k:
  kmeans = KMeans(n_clusters = element, init = 'k-means++')
  kmeans.fit(scaled_data)
  inert.append(kmeans.inertia_)

elbow_curve = pd.DataFrame({"x" : k, "y" : inert})
st.line_chart(elbow_curve)

st.write("""The number of clusters according to the elbow method is the whole number that occurs at the elbow of the graph.
            Based on this graph it would be optimal somewhere between 3 and 5.
            Now we can start the k-means clustering using  4 initial centroids and using k-means++ to find the position of the initial centroids.""")

# clusters = st.slider("Number of Clusters", min_value = 1, max_value = 10)

kmeans = KMeans(n_clusters = 4, init="k-means++")
kmeans.fit(scaled_data)
labels = kmeans.labels_
data['Cluster'] = labels  # Ensure the column name matches 'Cluster'

scatter_plot = alt.Chart(data).mark_circle(size=100).encode(
    x='Age',
    y='Spending Score (1-100)',
    color='Cluster:N',  # Matches the column name in the DataFrame
    tooltip=['Age', 'Spending Score (1-100)', 'Cluster']
).properties(
    title='Age vs Spending Score by Cluster',
    width=1000, 
    height=600
)

# Display the plot in Streamlit
st.altair_chart(scatter_plot, use_container_width=True)

scatter_plot = alt.Chart(data).mark_circle(size=100).encode(
    x='Gender',
    y='Spending Score (1-100)',
    color='Cluster:N',  # Matches the column name in the DataFrame
    tooltip=['Age', 'Spending Score (1-100)', 'Cluster']
).properties(
    title='Age vs Spending Score by Cluster',
    width=1000, 
    height=600
)

st.altair_chart(scatter_plot, use_container_width=True)

scatter_plot = alt.Chart(data).mark_circle(size=100).encode(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Cluster:N',  # Matches the column name in the DataFrame
    tooltip=['Age', 'Spending Score (1-100)', 'Cluster']
).properties(
    title='Age vs Spending Score by Cluster',
    width=1000, 
    height=600
)

st.altair_chart(scatter_plot, use_container_width=True)

# plt.figure(figsize=(8,6))
# plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=scaled_data['cluster'], cmap = 'plasma')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score')
# plt.title('Spending Score vs Annual Income (k$)')
# plt.colorbar(label='Cluster')
# plt.show()
