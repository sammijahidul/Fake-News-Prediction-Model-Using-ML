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

# Start Stemming process (reducing a word to its root word)
porter_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [porter_stem.stem(word) for word in stemmed_content if not
                       word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
news_data['content'] = news_data['content'].apply(stemming)
print(news_data['content'])

# Separating the data and label
X = news_data['content'].values
Y = news_data['label'].values

# Converting textual to numerical data 
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)

# Splitting the dataset to training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify = Y, random_state = 2)


### Data pre-processing part is done and now gonna start to build the ml model using logistic regression algorithm

# Training the Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluation the Model 
# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy for training dataset", training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy for test dataset", test_data_accuracy)

# Making a Predictive System 
X_new = X_test[0]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print("The news is Real")
else:
    print("The news is fake")
    







