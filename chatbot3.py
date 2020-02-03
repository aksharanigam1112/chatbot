import nltk
import string
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('wordnet')

text = "Ananlytics India Magazine (AIM) is India's no.1 platform on analytics, data science and big data"

text = text.lower()
sentences = nltk.sent_tokenize(text)
words = nltk.word_tokenize(text)

lemmer = nltk.stem.WordNetLemmatizer()

def lemtokens(words):
    return [lemmer.lemmatize(token, 'v') for token in words if token not in set(stopwords.words('english'))]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
print(remove_punct_dict)

def LemNormalize(text):
   return lemtokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def response(user_response):
        sentences.append(user_response)
        cv = CountVectorizer(max_features = 50, tokenizer = LemNormalize, analyzer = 'word')
        X = cv.fit_transform(sentences)
        vals_cv = cosine_similarity(X[-1], X)
        indx_of_most_similar_sentence = vals_cv.argsort()[0][-2] #sorting the indexes based on increasing similarity
        flat_vals_cv = vals_cv.flatten()
        flat_vals_cv.sort()
        highest_similarity = flat_vals_cv[-2] # required tfidf = most similar to 4

        if(highest_similarity == 0):
              robo_response = "I am sorry! I don't understand you"
              return robo_response
        else:
              robo_response = sentences[indx_of_most_similar_sentence]
              return robo_response

exit_codes = ['bye', 'see you', 'c ya', 'exit']
flag=True
print("Hi! Im a Chatty !")

while(flag==True):
        user_response = input("User:")
        if user_response.lower() not in exit_codes:
            user_response = user_response.lower()
            print("chatty :", response(user_response))
            sentences.remove(user_response)
            print('\nDo you want to continue ? (yes/no)')
            user_response = input("User-:yes/no? ")

            if user_response.lower() == 'no' or user_response.lower() == 'NO' or user_response.lower() in exit_codes :
                print('Bye!!')
                flag=False

        else :
                print('Bye!!')
                flag=False