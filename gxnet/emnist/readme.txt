This dataset is derived from kaggle's handwritten-alphabets dataset. 
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format/data

The original dataset has a total of 372451 data items. Two datasets, train and test, were selected from this dataset.

Select the test dataset first, and select up to 300 items for each label type. In the train dataset, select up to 10000 items for each label type. Since there is not enough data for each type of label in the original dataset, the amount of data for train and test is less than expected.

These are two python script to process the original dataset:
1) csv2mnist.py: convert csv to mnist format, to speed up loading dataset
2) splitmnist.py: split original dataset to train and test dataset

