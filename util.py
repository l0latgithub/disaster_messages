from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Tokernizer has to put into a separate library, otherwise pickle model could not loaded appropriately
# A former udacity student has a post on stackoverflow.com to discuss it, although I could not find the link
# The Tokernizer structure was also from the former udacity student. Sorry just cannot find the link now.
# Best post dicussses the issue: https://rebeccabilbro.github.io/module-main-has-no-attribute/
class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    
    def tokenize(self, X):
            import re
            # Use regex to find all urls and replace them to be a constant string
            url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            text = X
            detected_urls = re.findall(url_regex, text)

            for url in detected_urls:
                text = text.replace(url, "urlplaceholder")

            # Use NLTK word tokenizer and Lemmatizer to tokenize and lemmatize the messges
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()

            # Tokens are normalized to lower case and remove leading/trailing spaces
            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)
            
            # remove stopwords
            STOPWORDS = list(set(stopwords.words('english')))
            clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]

            return clean_tokens
        
    def transform(self, X):
        def tokenize(text):
            import re
            # Use regex to find all urls and replace them to be a constant string
            url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            detected_urls = re.findall(url_regex, text)

            for url in detected_urls:
                text = text.replace(url, "urlplaceholder")

            # Use NLTK word tokenizer and Lemmatizer to tokenize and lemmatize the messges
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()

            # Tokens are normalized to lower case and remove leading/trailing spaces
            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok).lower().strip()
                clean_tokens.append(clean_tok)
            
            # remove stopwords
            STOPWORDS = list(set(stopwords.words('english')))
            clean_tokens = [token for token in clean_tokens if token not in STOPWORDS]

            return " ".join(clean_tokens)
        return pd.Series(X).apply(tokenize).values

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    A customerized transformer to detect leading verb for the messages
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(Tokenizer().tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
