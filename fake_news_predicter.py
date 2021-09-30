# Importing neccessary library & packages to create the model
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# Printing the stopwords list here that won't create a value during training the model
print(stopwords.words('english'))

### Pre-processing the data starts from here 

# Importing the dataset & Checking the dataset features 
news_data = pd.read_csv('train.csv')
news_data.shape
news_data.head()

# Checking the number of missing values in the dataset
news_data.isnull().sum()

# Because we have a large dataset that's why we are going to replacing the null values with empty string
news_data = news_data.fillna('')

# Merging the author name and news title
news_data['content'] = news_data['author']+ ' ' +news_data['title']
print(news_data['content'])

# Separating the data & label
X = news_data.drop(columns='label', axis=1)
Y = news_data['label']



