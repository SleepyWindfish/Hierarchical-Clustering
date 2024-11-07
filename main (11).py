import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import linear_model
#Here we will use hierarchical clustering to group data points and visualize the clustering using both a dendrogram and a scatter plot. 
"""
x=[4,5,10,4,3,11,14,6,10,12]
y=[21,19,24,17,16,25,24,22,21,21]"""

X=np.array([3.78,2.44,2.09,0.14,1.72,1.65,4.92,4.37,4.96,4.52,3.69,5.88]).reshape(-1,1)
y=np.array([0,0,0,0,0,0,1,1,1,1,1,1,])

logr=linear_model.LogisticRegression()
logr.fit(X,y)

#Predict if tumor is cancerous where the size is 3.46mm
predicted=logr.predict(np.array([3.46]).reshape(-1,1))
print(predicted)


#zip Combines both data together (4,21)(5,19)(10,24)...
"""
data=list(zip(x,y))
linkage_data=linkage(data,method='ward',metric='euclidean')
dendrogram(linkage_data)
#plt.scatter(x,y)"""
plt.show()
