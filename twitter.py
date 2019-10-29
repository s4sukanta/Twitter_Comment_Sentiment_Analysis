#Sentiment Analysis on Twitter Comments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.stem import  WordNetLemmatizer, SnowballStemmer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB,ComplementNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the Train dataset
df=pd.read_csv("D:/DBA/project/Sentiment Analysis on Twitter_Comments/train.csv")
df.head(10)

#Count the number of values of each Sentiment
df.label.value_counts()

#Show the rows which has "Negative" Sentiment
df[df.label == 0]
#Show the rows which has "Positive" Sentiment
df[df.label == 1]


#Null value checking
df.isnull().sum()

# Text preprocessing (cleaning the reviews, tokenize and lemmatize the data)
def pre_process(text):
    text=text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    tokens=nltk.word_tokenize(text)  # tokenize
    wnl = WordNetLemmatizer()
    L=[wnl.lemmatize(w) for w in tokens]
    text=" ".join(L)
    return text

#Assign the dataset in input and output variable 
X = df['tweet'].apply(pre_process)
X.head(10)
y = df['label']
y.head(10)

# Split the data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Build the Model 
model1=make_pipeline(CountVectorizer(binary=True,stop_words="english"),BernoulliNB())
model2=make_pipeline(CountVectorizer(binary=False,stop_words="english"),MultinomialNB())
model3=make_pipeline(TfidfVectorizer(stop_words="english"),MultinomialNB())

# Fit the Model to the Train Dataset
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Predict y of Test Dataset
y_pred_1 = model1.predict(X_test)
y_pred_2 = model2.predict(X_test)
y_pred_3 = model3.predict(X_test)

# Confusion matrix of y_test and Model1 prediction result
mat= confusion_matrix(y_test,y_pred_1)
sns.heatmap(mat,annot=True,cbar=False,fmt='d')
plt.ylabel('True class')
plt.xlabel('Predicted class')

# overall accuracy
print("Model1 Score: ", model1.score(X_test,y_test).round(3))
print("Model2 Score: ",model2.score(X_test,y_test).round(3))
print("Model3 Score: ",model3.score(X_test,y_test).round(3))

""" Train the second model on the entire training data set 
 and use this model to predict unknown test data"""
model4=make_pipeline(CountVectorizer(binary=False,stop_words="english"),MultinomialNB())
model4.fit(X,y)

# load the test data
df1=pd.read_csv("D:/DBA/project/Sentiment Analysis on Twitter_Comments/test.csv")
df1.head(10)

#Cleansing the test dataset
Xtest=df1.tweet.apply(pre_process)
Xtest.head()

#Now make Prediction on Test dataset
predicted_sentiment=model4.predict(Xtest)
predicted_sentiment

#Now We create a DataFrame where we store the result as Sentiment analysis output
df2 = pd.DataFrame(predicted_sentiment,columns=['label'])
df2.head(10)

#Take the "id" column from test dataset and put our prediction result according to it. 
final_result = df2.join(df1['id']).iloc[:,::-1]
final_result.head(10)

#Write the output to the csv file.
final_result.to_csv("D:/DBA/project/Sentiment Analysis on Twitter_Comments/final_submission.csv", index=False)
