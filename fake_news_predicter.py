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

# Importing the dataset
news_data = pd.read_csv('train.csv')

