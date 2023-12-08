import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
import NLPmodule
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

POS_to_exclude = ['INTJ', 'PRCL', 'CONJ', 'NPRO', 'PREP']
STOP_WORDS = False

def morphed_sentence(sentence, stop_words=False):
    new_sentence = ''
    if stop_words:
        for word in [ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in sentence.split()]:
            if not(morph.parse(word)[0].tag.POS in POS_to_exclude): new_sentence += f"{word} "
    else:
        for word in [ morph.parse(NLPmodule.punctuation_marks_split(word))[0].normal_form  for word in sentence.split()]:
            new_sentence += f"{word} "
    return new_sentence

def parse_xml_to_dict(xml_string):
    root = ET.fromstring(xml_string)
    return parse_element(root)

def parse_element(element):
    result = {}

    # Обработка атрибутов элемента
    result.update(element.attrib)

    # Обработка дочерних элементов
    for child in element:
        child_data = parse_element(child)
        if child.tag in result:
            # Если у элемента уже есть значение в словаре,
            # превращаем его в список значений
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data

    # Обработка текстового содержимого элемента
    if element.text:
        result['text'] = element.text.strip()

    return result

def micro_macro_f(prediction, Y_test):
    true_pred, TP0, TPN, TPP = 0, 0, 0, 0
    FN0, FNN, FNP = 0, 0, 0
    for i,el in enumerate(list(prediction)):
        if el == Y_test[i]: true_pred+=1
        if el == Y_test[i] == '0': TP0+=1
        if el == Y_test[i] == '-': TPN+=1
        if el == Y_test[i] == '+': TPP+=1
        if Y_test[i] == '0' != el: FN0+=1
        if Y_test[i] == '-' != el: FNN+=1
        if Y_test[i] == '+' != el: FNP+=1    
    FP0 = list(prediction).count('0') - TP0
    FPN = list(prediction).count('-') - TPN
    FPP = list(prediction).count('+') - TPP

    P0 = TP0 / (TP0 + FP0)
    R0 = TP0 / (TP0 + FN0)
    PN = TPN / (TPN + FPN)
    RN = TPN / (TPN + FNN)
    P0 = TPP / (TPP + FPP)
    RP = TPP / (TPP + FNP)

    F0 = 2 * P0 * R0 / (P0 + R0)
    FN = 2 * PN * RN / (PN + RN)
    FP = 2 * FPP * RP / (FPP + RP)

    MacroF = (F0 + FN + FP) / 3
    TPsum = TP0 + TPN + TPP
    FPsum = FP0 + FPN + FPP
    FNsum = FN0 + FNN + FNP
    Psum = TPsum / (TPsum + FPsum)
    Rsum = TPsum / (TPsum + FNsum)
    MicroF = 2 * Psum * Rsum / (Psum + Rsum)
    return MicroF, MacroF

xml_filepath = 'train/news_eval_train.xml'
with open(xml_filepath, 'r') as file:
    xml_data = file.read()
xml_dict = parse_xml_to_dict(xml_data)
tags = ('+','-','0')
dict_train = { morphed_sentence(xml_dict['sentence'][id]['speech']['text'], STOP_WORDS) : xml_dict['sentence'][id]['evaluation']['text']
               for id in range(len(xml_dict['sentence'])) if xml_dict['sentence'][id]['evaluation']['text'] in tags}

xml_filepath = 'test/news_eval_test.xml'
with open(xml_filepath, 'r') as file:
    xml_data = file.read()
xml_dict = parse_xml_to_dict(xml_data)
dict_test = { morphed_sentence(xml_dict['sentence'][id]['speech']['text'], STOP_WORDS) : xml_dict['sentence'][id]['evaluation']['text']
               for id in range(len(xml_dict['sentence'])) if xml_dict['sentence'][id]['evaluation']['text'] in tags}

vectorizer = CountVectorizer()
#векторизация обучающих и тестовых данных
corpus = list(dict_train.keys()) + (list(dict_test.keys()))
X_train = vectorizer.fit_transform(corpus).toarray()[0:len(list(dict_train.keys()))]
Y_train = list(dict_train.values())
X_test = vectorizer.fit_transform(corpus).toarray()[len(list(dict_train.keys())):]
Y_test = list(dict_test.values())

#переход к векторизации tf idf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
counts = vectorizer.fit_transform(corpus).toarray()
transformed = transformer.fit_transform(counts).toarray()
X_train_tfidf = transformed[0:len(list(dict_train.keys()))]
X_test_tfidf = transformed[len(list(dict_train.keys())):]

#переход к векторизации булевской
X_train_bool = [list(map(bool, arr)) for arr in X_train ]
X_test_bool = [list(map(bool, arr)) for arr in X_test ]

from sklearn import svm
#from sklearn.linear_model import SGDClassifier
#from sklearn import tree
clf = svm.SVC()
##clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10)
#clf = tree.DecisionTreeClassifier()

#частотная векторизация
clf.fit(X_train, Y_train)
prediction1 = clf.predict(X_test)
print("frequencies:  ", micro_macro_f(prediction1, Y_test))

#tf.idf векторизация
clf.fit(X_train_tfidf, Y_train)
prediction2 = clf.predict(X_test_tfidf)
print("tf.idf:  ", micro_macro_f(prediction2, Y_test))

#булевская векторизация
clf.fit(X_train_bool, Y_train)
prediction3 = clf.predict(X_test)
print("bool:  ", micro_macro_f(prediction3, Y_test))