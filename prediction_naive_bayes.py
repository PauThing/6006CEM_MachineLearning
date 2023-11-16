from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

# load the dataset
tweets_df = pd.read_csv('Tweets.csv')

# remove quotes at the beginning and end of the text field
tweets_df['text'] = tweets_df['text'].str.strip('\"')
tweets_df['selected_text'] = tweets_df['selected_text'].str.strip('\"')

# drop the 'textID' and 'selected_text' columns
tweets_df.drop(['textID'], axis=1, inplace=True)

# reset the index of DataFrame after dropping columns
tweets_df.reset_index(drop=True, inplace=True)

# assign the column name 'sentiment' to the variable
target = 'sentiment'

# create a deep copy of the DataFrame df 
df = tweets_df.copy(deep=True)

# split the text into individual words or tokens
tweets_df['text_tokens'] = tweets_df['selected_text'].apply(word_tokenize)

# convert the text data into numerical vectors using TF-IDF representation
# map sentiment labels to numerical values for classification
# create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# remove common words that don't carry much meaning
stop_words = set(stopwords.words('english'))
tweets_df['text_tokens'] = tweets_df['text_tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])

# reduces words to their base or root form
stemmer = PorterStemmer()
tweets_df['text_tokens'] = tweets_df['text_tokens'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# fit and transform the text data
X_tfidf = tfidf_vectorizer.fit_transform(tweets_df['text_tokens'].apply(lambda x: ' '.join(x)))

# convert the sentiment labels to numerical values
label_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
y = tweets_df[target].map(label_mapping)

# display the positive words as WordCloud image
# text = " ".join(i for i in tweets_df[tweets_df[target]=='positive']['selected_text'])
# wordcloud = WordCloud( background_color="white").generate(text)

# plt.figure( figsize=(15,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('wordcloud for positive words')
# plt.show()

# display the negative words as WordCloud image
# text = " ".join(i for i in tweets_df[tweets_df[target]=='negative']['selected_text'])
# wordcloud = WordCloud( background_color="white").generate(text)

# plt.figure( figsize=(15,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('wordcloud for negative words')
# plt.show()

# display the neutral words as WordCloud image
# text = " ".join(i for i in tweets_df[tweets_df[target]=='neutral']['selected_text'])
# wordcloud = WordCloud( background_color="white").generate(text)

# plt.figure( figsize=(15,10))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title('wordcloud for neutral words')
# plt.show()

# overall distribution of positive, negative and neutral sentiments
# plt.pie(tweets_df[target].value_counts(), labels=['Neutral','Positive','Negative'], counterclock=False, shadow=True, 
#         explode=[0,0,0.08], autopct='%1.1f%%', radius=1, startangle=0)
# plt.show()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# create and train a Naive Bayes (Multinomial) model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

# predictions on the test set
y_pred_naive_bayes = naive_bayes_model.predict(X_test)

# evaluate the Naive Bayes (Multinomial) model
accuracy_naive_bayes = accuracy_score(y_test, y_pred_naive_bayes)
classification_report_naive_bayes = classification_report(y_test, y_pred_naive_bayes)

print(f"Naive Bayes Accuracy: {accuracy_naive_bayes:.4f}")
print("Naive Bayes Classification Report:\n", classification_report_naive_bayes)

# test with user input
# user input
user_input = input("Enter a tweet: ")

# preprocess the user input
user_input_tokens = word_tokenize(user_input)
user_input_cleaned = ' '.join([word.lower() for word in user_input_tokens if word.isalpha()])

# vectorize the user input using the same tfidf_vectorizer
user_input_vectorized = tfidf_vectorizer.transform([user_input_cleaned])

# prediction on the user input
predicted_sentiment = naive_bayes_model.predict(user_input_vectorized)

# define a mapping between numeric labels and text labels
label_mapping_reverse = {1: 'positive', -1: 'negative', 0: 'neutral'}

# map the predicted numeric sentiment to text sentiment
predicted_sentiment_text = label_mapping_reverse[predicted_sentiment[0]]

# print the predicted sentiment
print("Predicted Sentiment:", predicted_sentiment[0], " (", predicted_sentiment_text, ")")