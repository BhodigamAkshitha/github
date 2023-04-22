#!/usr/bin/env python
# coding: utf-8

# ## Step 4:Exploring Data
# 

# In[2]:


import pandas as pd
import numpy as np

mcdonalds=pd.read_csv('mcdonalds.csv')
print(list(mcdonalds.columns))


# In[3]:


mcdonalds.shape


# In[4]:


mcdonalds.head(3)


# In[4]:


MD_x =(mcdonalds.iloc[:, 0:11])
MD_x=(MD_x == "yes").astype(int)

col_means = np.round(np.mean(MD_x, axis=0), 2)
print (col_means)


# In[5]:


from sklearn.decomposition import PCA
import pandas as pd

MD_x = mcdonalds.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
MD_pca = PCA().fit(MD_x)

# Print the explained variance and the loadings of the principal components
print("Explained Variance Ratio:")
print(MD_pca.explained_variance_ratio_)

n_components = MD_pca.n_components_
summary_table = pd.DataFrame({
    "PC": ["PC" + str(i+1) for i in range(n_components)],
    "Standard deviation": MD_pca.explained_variance_**0.5,
    "Proportion of Variance": MD_pca.explained_variance_ratio_,
    "Cumulative Proportion": np.cumsum(MD_pca.explained_variance_ratio_)
})
print("\nSummary Table:")
print(summary_table)


# In[6]:


MD_x = mcdonalds.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
MD_pca = PCA().fit(MD_x)

# Print the principal component analysis with one decimal place

print(pd.DataFrame({
    "Standard Deviation": MD_pca.explained_variance_**0.5,
    
}).round(1))


# In[7]:


# Get the loading matrix
loading_matrix = MD_pca.components_.T

# Convert the loading matrix into a DataFrame with variable names as row names
loading_matrix_df = pd.DataFrame(loading_matrix, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'], 
                                 index=['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting'])

# Round the values in the DataFrame to 3 decimal places
loading_matrix_df = loading_matrix_df.round(3)

# Print the loading matrix
print(loading_matrix_df)


# In[8]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming your data is stored in a numpy array named `data`
pca = PCA(n_components=2)
pca.fit(MD_x)
transformed_data = pca.transform(MD_x)

plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

proj_axes = pca.components_.T[:, :2]  # First two principal components
print(proj_axes)


# ## Step 5:Extracting Segments

# In[9]:


import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# set random seed
np.random.seed(1234)

# perform PCA on data
pca = PCA(n_components=2)
MD_pca = pca.fit_transform(MD_x)

# perform KMeans clustering on PCA-reduced data
n_clusters = range(2, 9)
kmeans_models = [KMeans(n_clusters=k, n_init=10) for k in n_clusters]
kmeans_results = [kmeans_models[i].fit(MD_pca) for i in range(len(kmeans_models))]

# relabel the KMeans results
MD_km28 = [kmeans_results[i].labels_ for i in range(len(kmeans_results))]


# In[10]:


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(1234)
inertias = []
for n_clusters in range(2, 9):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(MD_x)
    inertias.append(kmeans.inertia_)
plt.plot(range(2, 9), inertias, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Inertia')
plt.show()


# In[11]:


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

mcdonalds=pd.read_csv('mcdonalds.csv')

# Assuming MD.x is a numpy array or a pandas dataframe
MD_x = mcdonalds.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)
k = 4
kmeans_model = KMeans(n_clusters=k,n_init='auto', random_state=0).fit(MD_x)
MD_k4 = kmeans_model.labels_
MD_r4 = kmeans_model.predict(MD_x)


# In[12]:


import matplotlib.pyplot as plt

plt.plot(MD_r4, 'o-')
plt.ylim(0,1)
plt.xlabel("segment number")
plt.ylabel("segment stability")
plt.show()


# ### Using Mixtures of Distributions

# In[13]:


import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
import pandas as pd
import numpy as np

mcdonalds=pd.read_csv('mcdonalds.csv')
np.random.seed(1234)
MD_x = mcdonalds.iloc[:, 0:11].values
MD_x = (MD_x == "Yes").astype(int)


MD_x = preprocessing.scale(MD_x) # Standardize the data

BIC = []
for k in range(2, 9):
    model = GaussianMixture(n_components=k, covariance_type='full')
    model.fit(MD_x)
    BIC.append(model.bic(MD_x))

MD_m28 = {'BIC': BIC}
print(MD_m28)


# In[14]:


## Using Mixtures of Regression Models


# In[15]:


freq_table = mcdonalds['Like'].value_counts()
reversed_table = freq_table.iloc[::-1]
print(reversed_table)


# In[16]:


mcdonalds['Like.n'] = 6 - mcdonalds['Like'].apply(lambda x: int(x.split('!')[1]) if '!' in x else int(x))
print(mcdonalds['Like.n'].value_counts())


# In[17]:


f = " + ".join(mcdonalds.columns[:11])
f = "Like.n ~ " + f
print (f)


# ## Step 6: Proﬁling Segments
# 

# In[18]:


from sklearn.cluster import KMeans
import numpy as np
np.random.seed(1234)
kmeans = KMeans(random_state=1234)
n_clusters = range(2,9)
inertias = []
for k in n_clusters:
    kmeans.set_params(n_clusters=k,n_init='auto')
    kmeans.fit(MD_x)
    inertias.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(n_clusters, inertias, '-o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
n_clusters = 3
kmeans.set_params(n_clusters=n_clusters)
MD_km28 = kmeans.fit_predict(MD_x)


# In[19]:


import matplotlib.pyplot as plt
MD_km28 = kmeans.fit_predict(MD_x)
# plot the cluster assignments
plt.scatter(range(len(MD_km28)), MD_km28)
plt.xlabel('Number of segments')
plt.ylabel('Cluster assignment')
plt.show()


# In[20]:


MD_km28 = kmeans.fit_predict(MD_x)
MD_k4 = MD_km28[3]


# ### Step 7: Describing Segments

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA



# convert "mcdonalds" to a numpy array
MD_x = mcdonalds.iloc[:, :11].apply(lambda x: x == "Yes").astype(int).to_numpy()

# perform PCA
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# perform k-means with k=4
kmeans = KMeans(n_clusters=4,n_init='auto', random_state=1234)
kmeans.fit(MD_x)
k4 = kmeans.predict(MD_x)

# create a DataFrame with the clustering labels and the "Like" variable
data = pd.DataFrame({"Segment": k4, "Like": mcdonalds.Like})

# create the mosaic plot
mosaic(data, ["Segment", "Like"], title="", axes_label=True, gap=0.01)

# show the plot
plt.show()


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

plt.rcParams["figure.figsize"] = (8, 6)

mosaic_data = pd.crosstab(index=mcdonalds["Gender"], columns=k4)
mosaic(mosaic_data.stack(), gap=0.05, title="Gender by Segment", properties=lambda key: {'color': 'gray' if key[1] == 3 else 'lightgray'})
plt.xlabel("Gender")
plt.ylabel("Segment number")
plt.show()


# ### Step 8: Selecting (the) Target Segment(s)

# In[23]:


female = mcdonalds['Gender'].eq('Female').groupby(k4).mean()
print(female)


# In[24]:


like = mcdonalds.groupby(k4)['Like.n'].mean()
print(like)


# In[ ]:





# In[ ]:




