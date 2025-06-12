# Tokenizacion
import pandas as pd
from sklearn.model_selection import train_test_split # Esto nos permite dividir la base de datos en un conjunto de datos de entrenamiento y en otro conjunto para probar el modelo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

df = pd.read_csv("df_total.csv", encoding="UTF-8")
df.info()

# Separamos los datos en variables de entrada y etiquetas
X = df["news"]
y = df["Type"]

# El test size nos dice con cuanto porcentaje vamos a probar los datos, en este caso entrenaremos el modelo con el 80% y lo testearemos con el 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)
print(f"original: {metrics.accuracy_score(y_test, y_pred)}")

# Stemming: Recorta los sufijos de las palabras para obtener una raíz común (aunque no siempre sea una palabra real).
# Quitarle los sufijos a las palabras para disminuir cantidad de dimensiones.
# Ej: caminar, caminando, camino = cami

stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    tokens = word_tokenize(text.lower())
    stems = [stemmer.stem(token) for token in tokens if token.isalpha()]
    return " ".join(stems)

df["news_stemmer"] = df["news"].apply(tokenize_and_stem)

# Hacemos lo mismo que antes pero ahora con stemmer
X = df["news_stemmer"]
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)
print(f"stemming: {metrics.accuracy_score(y_test, y_pred)}")

# Lemmatization: Reduce las palabras a su forma canónica (lema), teniendo en cuenta la gramática y el contexto.
# Mas preciso que un stemmer
# Ej: corriendo, correr, corria = correr
nlp = spacy.load("es_core_news_sm")

# Cargar stopwords desde NLTK
stopwords_nltk = set(stopwords.words('spanish'))

def lemmatize_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords_nltk]
    return " ".join(lemmas)

df["news_lemma"] = df["news"].apply(lemmatize_text)

# Hacemos lo mismo que antes pero ahora con lemmatizacion
X = df["news_lemma"]
y = df["Type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)
print(f"lemma: {metrics.accuracy_score(y_test, y_pred)}")