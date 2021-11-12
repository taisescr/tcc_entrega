
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def vetorizaTFIDF(vocabulario, texto):
    vectorizer_tfidf = TfidfVectorizer(lowercase=True, analyzer='word', binary=False, ngram_range=(1,3))
    vectorizer_tfidf.fit_transform(vocabulario.tolist()) 
    return vectorizer_tfidf.transform(texto)    

def seleciona(vocabulario, x, y, perc):
    k = round(max(vocabulario.apply(len))*perc)
    return SelectKBest(chi2, k=k).fit_transform(x, y)