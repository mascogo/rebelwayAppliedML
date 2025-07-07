import pandas as pd
from pprint import pformat
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

dataset_csv = r'C:\Cursos_Rebelway\ML_for_3D_and_VFX_MAY2025\myDataSets\nlp_emotions\combined_emotion.csv'
df = pd.read_csv(dataset_csv)

df['sentence'] = df['sentence'].str.lower()

X_train, X_test, y_train, y_test = train_test_split(df['sentence'],df['emotion'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer(stop_words="english")
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("X_train ({}): \n{}\n".format(len(X_train), pformat(X_train)))
print("X_train_vectorized: \n{}\n".format(pformat(X_train_vectorized.toarray())))
print("X_train_vectorized ({}): \n{}\n".format(type(X_train_vectorized), pformat(X_train_vectorized)))

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

y_pred = nb_classifier.predict(X_test_vectorized)
print("Accuracy: ", accuracy_score(y_test,y_pred))
print("Classification report: ", classification_report(y_test,y_pred))

new_sentences = ["I'm so happy today!", "This is terrifying.", "This is wong.", "what a terrific issue."]
new_sentences_vectorized = vectorizer.transform(new_sentences)

print("new_sentences_vectorized: \n{}".format(pformat(len(list(new_sentences_vectorized.toarray()[0])))))
predictions = nb_classifier.predict(new_sentences_vectorized)
print(new_sentences)
print("Predictions: ", predictions)