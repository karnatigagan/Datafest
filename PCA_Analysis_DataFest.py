#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

# # Load the dataset
# dataset_path = 'media_views.csv' 
# data = pd.read_csv(dataset_path)

# # Display the first few rows of the dataset to understand its structure
# data.head()

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.impute import SimpleImputer

# # Selecting numerical columns and dropping rows with missing values in these columns
# numerical_columns = ['chapter_number', 'section_number', 'access_count', 'proportion_video', 'proportion_time']
# numerical_data = [numerical_columns]

# # Imputing missing values with the mean for each column
# imputer = SimpleImputer(strategy='mean')
# imputed_data = imputer.fit_transform(numerical_data)

# # Standardizing the data
# scaler = StandardScaler()
# standardized_data = scaler.fit_transform(imputed_data)

# # Applying PCA
# pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
# pca_result = pca.fit_transform(standardized_data)

# # Output the explained variance by each component
# explained_variance = pca.explained_variance_ratio_

# pca_result[:5], explained_variance


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the dataset
dataset_path = 'media_views.csv'
data = pd.read_csv(dataset_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Selecting numerical columns and handling missing values by imputation
numerical_columns = ['chapter_number', 'section_number', 'access_count', 'proportion_video', 'proportion_time']
numerical_data = data[numerical_columns]

# Imputing missing values with the mean for each column
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(numerical_data)

# Standardizing the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(imputed_data)

# Applying PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization purposes
pca_result = pca.fit_transform(standardized_data)

# Output the explained variance by each component
explained_variance = pca.explained_variance_ratio_

# Displaying the first five results of PCA and the explained variance
print(pca_result[:5], explained_variance)


# In[13]:


import matplotlib.pyplot as plt

# Assuming pca_result is the result from PCA transformation
# and we have already computed pca_result using the previous steps

# Convert the PCA result into a DataFrame for easier plotting
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Create a scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])

plt.title('PCA Results: First Two Principal Components')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()


# In[19]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('media_views.csv')

# Assuming that the relevant features are the numerical ones
features = ['chapter_number', 'section_number', 'access_count', 'proportion_video', 'proportion_time']
X = data[features]

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Apply PCA
pca = PCA(n_components=5)  # Adjust the number of components you want to use
X_pca = pca.fit_transform(X_scaled)

# Plot the explained variance to see how many components you should choose
plt.figure(figsize=(8, 6))
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') # for each component
plt.title('Explained Variance')
plt.show()

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First Two Principal Components')
plt.show()


# In[20]:


# Apply PCA
pca = PCA(n_components=5)  # Using all 5 components since there are 5 features
X_pca = pca.fit_transform(X_scaled)

# Extract the loadings (components)
loadings = pca.components_

# Print the loadings for each principal component
for i, component in enumerate(loadings):
    print(f"Principal Component {i+1}:")
    for j, loading in enumerate(component):
        print(f"  {features[j]}: {loading:.3f}")
    print()

# Identify and print the most influential variable for each principal component
most_influential_features = [features[loading.argmax()] for loading in loadings]
print("Most influential variable for each principal component:")
for i, feature_name in enumerate(most_influential_features):
    print(f"  PC{i+1}: {feature_name}")


# In[ ]:




