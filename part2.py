#arrays
import pandas as pd
import numpy as np
#machine learning
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#visualization
import matplotlib.pyplot as plt

dataset = load_iris() #using the iris dataset
data = dataset.data

#colors for visualization
LABEL_COLOR_MAP = {0 : 'red',
                   1 : 'green',
                   2 : 'blue',
                   }

#eda - show ground truth
#find linearly separable data points.
eda = False #set to True to do eda
if eda:
    print("showing plots to find linearly seperable values")
    labels = dataset.target
    colors = [LABEL_COLOR_MAP[i] for i in labels]
    
    for i in range(4):
        for j in range(i+1,4):
            x = np.array([data[:,i]])
            y = np.array([data[:,j]])
            plt.scatter(x,y,c=colors)
            plt.title("Columns: {0} and {1}".format(i,j))
            plt.show()
        
#columns 2 and 3 were chosen because they appeared to be linearly separable in the ground truth

#get x and y values
#make dataframe for kmeans
x = np.array([data[:,2]]).reshape((150,1))
y = np.array([data[:,3]]).reshape((150,1))
d = pd.DataFrame(np.append(x,y,1))

#elbow method - 3 looks best after running it
#this is to be expected since there are 3 classes.
print("printing silhouette scores to find a good value for k")
wcss = []
for nclusters in range(1,11):
    kmeans = KMeans(n_clusters=nclusters,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(d)
    labels = kmeans.predict(d)
    wcss.append(kmeans.inertia_)
    # Calculate the silhouette score for the clustering if more than 1 cluster
    if nclusters > 1:
        print("For n_clusters = {0} The average silhouette score is: {1}".format(nclusters,silhouette_score(d, labels)))

print("showing elbow method to find a good value for k")
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

#kmeans with k = 3
nclusters = 3 
seed = 0
mdl = KMeans(n_clusters=nclusters, random_state=seed).fit(d)

labels = mdl.predict(d)

colors = [LABEL_COLOR_MAP[i] for i in labels]

plt.scatter(x,y,c=colors)
plt.title('kmeans clustering where k = {0}'.format(nclusters))
plt.show()


