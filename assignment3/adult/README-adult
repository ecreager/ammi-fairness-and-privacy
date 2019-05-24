Information about Adult dataset

This is a often-used simple dataset for ML models. It contains data from the 1994 US census. It was downloaded from the UCI repo (https://archive.ics.uci.edu/ml/datasets/adult) which contains many other simple datasets.

Raw data from UCI repo:
adult.data - training set
adult.test - test set
adult.names - information about the data

We've done some pre-processing for you, so you can just load a Numpy array into your Python program.

Pre-processed data:
adult_train.npz - training data, 32561 examples by 113 attributes
adult_test.npz - testing data, 16281 examples
adult_headers.txt - a list of column names for the pre-processed dataset, indicating what each feature represents. The final entry is “income”, which is the label but is not in the features X.

You can load the data from the npz file with np.load(filename). This should return a dictionary with keys ‘x’, ‘y’, and ‘a’.

The pre-processing included removing some features and converting some features to one-hot representation. You may want to additionally whiten the features i.e. subtract the mean and divide by the standard deviation of each column.

The label you'll predict (income) and the sensitive attribute (sex), are both features in this dataset. Before learning to predict income (or sex), you'll need to remove it from the data.
