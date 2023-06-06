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



# Sample input dictionaries
data = [
    {'color': 'orange', 'size': 'small'},
    {'color': 'white', 'model': 'v1.01'},
    {'color': 'green', 'year': '2023'}
]



# Create an instance of CustomDictVectorizer
vectorizer = SampleDictVectorizer()

# Fit the data
vectorizer.fit(data)

# Transform the data
X_sparse = vectorizer.transform(data)

# Print the resulting sparse matrix
print(X_sparse)