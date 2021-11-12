
import nltk
from nltk.corpus import stopwords
import re
from unicodedata import normalize
from nltk.tokenize import RegexpTokenizer
import html
import spacy
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from libs import stopword 
import re, cgi
nlp = spacy.load('pt_core_news_sm')


def lemmatizar(texto):
    lemma = ""
    for token in nlp(texto):
        lemma += token.lemma_ + " "
    return lemma


def removeTag(texto):
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    semTags = tag_re.sub('', texto)
    return html.escape(semTags)

def limpaTexto(texto):
    tokenizer = RegexpTokenizer(r'\w+')
    texto = texto.lower()
    texto = removeTag(texto)
    texto = html.unescape(texto)
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'[^a-zA-Z_]',' ',texto)
    tokens = tokenizer.tokenize(texto)
    tokens = [palavra for palavra in tokens if len(palavra) >= 3]      
    tokens = stopword.remover_stopwords(tokens)
    texto = ' '.join(tokens)
    texto = lemmatizar(texto)
    return texto

def buscaVerbo(texto):
    verbos = []

    doc = nlp(texto)
    for token in doc:
        if token.pos_ =='VERB':
            verbos.append(str(token))
    soverbos = " ".join(e for e in verbos)
    return(soverbos)

def geraNuvem(textos,titulo):
    #Ementa
    textocompleto = " ".join(e for e in textos)
    wordcloud = WordCloud(max_font_size=100,width = 1520, height = 535).generate(textocompleto)
    plt.figure(figsize=(16,9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(titulo)
    plt.show()