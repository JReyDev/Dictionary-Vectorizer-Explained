# Dictionary-Vectorizer-Explained

#### This is for informational purposes only.
#### This is an explanation of the general implementation of the dictionary vectorizer in scikit-learn aka one hot encoding. The purpose of this vectorizer is feature extraction, to convert dictionary categorical features into a sparse matrix for machine learning algorithms.



<img src='https://github.com/JReyDev/Dictionary-Vectorizer-Explained/blob/main/images/homepageconversion.png'>


## Create a class to store our data
```
import numpy as np

class SampleDictVectorizer:
    def __init__(self):
        self.feature_indices_ = {}
        self.num_features_ = 0 


    def fit(self, data):

        for dictionary in data: 
            for feature in dictionary.keys():
                if feature not in self.feature_indices_: 
                    self.feature_indices_[feature] = self.num_features_
                    self.num_features_ += 1 



    def transform(self, data):

        num_samples = len(data) 
        X_sparse = np.zeros((num_samples, self.num_features_))

        for index, dictionary in enumerate(data):
            for feature in dictionary.keys():
                if feature in self.feature_indices_:
                    feature_index = self.feature_indices_[feature]
                    X_sparse[index, feature_index] = 1

        return X_sparse

```

## Storing features

```
self.feature_indices_ = {}
self.num_features_ = 0 
```
