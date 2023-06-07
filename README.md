# Dictionary-Vectorizer-Explained

#### This is for informational purposes only.
#### This is an explanation of the general implementation of the dictionary vectorizer in scikit-learn aka one hot encoding. The purpose of this vectorizer is feature extraction, to convert dictionary categorical features into a sparse matrix for machine learning algorithms.



<img src='https://github.com/JReyDev/Dictionary-Vectorizer-Explained/blob/main/images/homepageconversion.png'>

```

```
```
import numpy as np

class SampleDictVectorizer:
    def __init__(self):
        self.feature_indices_ = {}
        self.num_features_ = 0
```
