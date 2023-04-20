# Fake News Detection Using Logistic Regression 

![Screenshot (24)](https://user-images.githubusercontent.com/117622500/233439788-21be445d-01d5-41a7-8da2-cd4466b45b25.png)

Widespread belief in misinformation circulating online is a critical challenge for modern societies. According to an MIT study, falsehood diffuses significantly farther, faster, deeper, and more broadly than the truth, in all categories of information, and in many cases by an order of magnitude. This is done mainly by clickbaits which lure users and entice curiosity with flashy headlines or designs and trick them into clicking the links to increase the ad revenues. Hence it is crucial to determine the integrity of the information available on the internet in order to keep the prevalence of fake news in check.
So we’ll now try to build a simple Machine Learning Model using Logistic Regression to detect whether a news article is fake or not.

Logistic Regression
Before diving into the code let us revise the Logistic Regression concept. Logistic regression is a statistical analysis method to predict a binary outcome, such as yes or no(binary classification), based on prior observations of a data set. It is a Supervised statistical technique to find the probability of dependent variable. The graph shown below is a Sigmoid Function, which we also call as a Logit. This function converts the probabilities into binary values which could be further used for predictions.

![Screenshot (23)](https://user-images.githubusercontent.com/117622500/233444591-ef1ff479-9679-4b86-a2df-96e7e0c34203.png)

According to this graph, if we obtain the probability value to be less than 0.5, then it is considered to be of the Class 0 and if the value is more than 0.5

## Code
Now finally starting off with our code, you can either write it in your Jupyter Notebook or Google Colab or any other platform you like.
```
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Covering the importance of each library/module/function that we imported:

#### NumPy : It is a general-purpose array and matrices processing package.
#### Pandas : It allows us to perform various operations on datasets.
#### re : It is a built-in RegEx package, which can be used to work with Regular Expressions.
#### NLTK : It is a suite of libraries and programs for symbolic and statistical natural language processing (NLP).
#### nltk.corpus : This package defines a collection of corpus reader classes, which can be used to access the contents of a diverse set of corpora.
#### stopwords : The words which are generally filtered out before processing a natural language are called stop words. These are actually the most common words in any language (like articles, prepositions, pronouns, conjunctions, etc) and does not add much information to the text. (Example-and, of, are etc.)
#### PorterStemmer : A package to help us with stemming of words. (More about stemming in the Data Preprocessing section)
#### Sci-kit Learn (sklearn) : It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.
#### feature_extraction.text : It is used to extract features in a format supported by machine learning algorithms from datasets consisting of text.
#### TfidfVectorizer : It transforms text to feature vectors that can be used as input to estimator. (More about TfidfVectorizer in the Data Preprocessing section)
#### train_test_split : It is a function in Sklearn model selection for splitting data arrays into two subsets - for training data and for testing data.
#### LogisticRegression : A pretty self explanatory part of the code, used to import the Logistic Regression Classifier.
#### metrics and accuracy_score : To import Accuracy classification score from the metrics module.
## Loading the Dataset
I hope you have downloaded the dataset by now. Now you can load the dataset as,
```
data = pd.read_csv('fakenews.csv')
data.head()
```
Here I have renamed my csv file as fakenews.csv and saved it in the same folder as my jupyter notebook. If you saved your dataset and the jupyter notebook in 2 different folders, you can add the path of the dataset file as the prefix in the code as,
Here Label indicates whether a news article is fake or not, 0 denotes that it is Real and 1 denotes that it is Fake.

## Data Preprocessing
After importing our libraries and the dataset, it is important to preprocess the data before we train our ML model since there might be some anomalies and missing datapoints which might make our predictions a bit skewed from the actual values.

Now, we can check the size of the dataframe/table as it would decide whether we can drop the rows with null values without affecting the size of our dataset or not.
```
data.shape
```
This gives us (20800, 5) which means that we have 20800 number of entries and 5 columns (features).

Checking the total number of missing values in each of the columns.
```
data.isnull().sum()  
```
From this we can see that we will have to delete a minimum of 1957 lines to remove all the null values so it would be better to fill these null values with an empty string. For that we can use fillna.
```
df1 = data.fillna('')
```
After this step we no longer have any missing datapoints, you can check that using the isnull().sum()

Now, we’ll try to reduce those 5 columns to only 2 columns since it will be easier for us to train the model. For that we’ll combine the title and the author columns into one, naming it as content. We can drop the other columns as they don’t have much effect on determining whether the article is fake or not. This step will leave us with 2 columns - content and label.
```
df1['content'] = df1['author'] + ' ' + df1['title']
```
## Stemming

Now coming to the stemming part, it basically is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words.


Stemming example
Stemming of words might or might not end up with a root word with meaning, like in this example chang doesn’t mean change or anything as a matter of fact. For the root word to have meaning we use lemmatization. But for this project stemming works just fine.
```
stemmer = PorterStemmer()
```
We create a new Porter stemmer for us so that we can use the function without explicitly typing PorterStemmer() every time.
```
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content) #1
    stemmed_content = stemmed_content.lower() #2
    stemmed_content = stemmed_content.split() #3
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')] #4
    stemmed_content = ' '.join(stemmed_content) #5
    return stemmed_content #6
```
Okay, so let’s go in depth and see what this function actually does. I have numbered each line from 1 to 6 so that you can easily distinguish between different lines of code and understand each line’s use.

#1 First we use the re package and remove everything that is not a letter (lower or uppercase letters).

#2 We then convert every uppercase letter to a lower one.

#3 We then split the each sentence into a list of words.

#4 Then we use the stemmer and stem each word which exists in the column and remove every english stopword present in the list.

#5 We then join all these words which were present in the form of a list and convert them back into a sentence.

#6 Finally we return the stemmed_content which has been preprocessed.

Applying this function to our dataset,
```
df1['content'] = df1['content'].apply(stemming)
df1['content'].head()
```
Next step is to name our input and output features
```
X = df1.content.values
y = df1.label.values
```
Our last preprocessing step would be to transform our textual X to numerical so that our ML model can understand it and can work with it. This is where TfidfVectorizer comes into play. Here is a picture explaining it in brief,


Source: https://becominghuman.ai/word-vectorizing-and-statistical-meaning-of-tf-idf-d45f3142be63
To understand it in depth, visit this link.
```
X = TfidfVectorizer().fit_transform(X)
print(X)
```
The output of this code should like this,


Now that we have the X in our desired form, we can move onto the next step.

## Splitting the Dataset
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)
```
This means that we have divided our dataset into 80% as training set and 20% as test set. stratify = y implies that we have made sure that the division into train-test sets have around equal distribution of either classes (0 and 1 or Real and Fake). random_state = 2 will guarantee that the split will always be the same.

## Training the Model
Fitting the model to our dataset
```
model = LogisticRegression()
model.fit(X_train, y_train)
```
Now that we have trained it, let’s check the accuracy of our training set predictions,
```
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(X_train_prediction, y_train)
print(training_accuracy)
```
Training accuracy score
So I got about 98.66%, which is pretty good. Similarly for the test dataset.
```
X_test_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(X_test_prediction, y_test)
print(testing_accuracy)
```
###Test accuracy score
So test accuracy is also pretty good.

(Note: The score may differ for you if you make any changes to this code)

With this we have successfully trained our ML model!

Building a system
Finally to make this model useful we need to make a system. Taking a sample out of the test-set (I took the first sample),
```
X_sample = X_test[0]
```
Checking our prediction for this sample,
```
prediction = model.predict(X_sample)
if prediction == 0:
    print('The NEWS is Real!')
else:
    print('The NEWS is Fake!')
```
With this we have built a system as well. Now if you want to take it a step further, try inputing a textual sample and predict using that. You can now give yourself a pat on the back as you now know how to detect a Fake News article using Logistic Regression only!!!
