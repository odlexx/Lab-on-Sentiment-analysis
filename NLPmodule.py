from math import sqrt

# задаем множества знаков препинаний и цифр
punctuation_marks = (".", ",", "!", "?", "–", ":", "(", ")", "[", "]", "«", "»", "„", "“", ";", "-", "'", "—", "…", "-",
                     "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "–",'"')
digits = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")

# функция, которая удаляет из начала и конца слова знаки препинания (оставляет числа и кдиничные знаки препинания, например тире меду словами)
def punctuation_marks_split(word):
    length = len(word)
    if word.isdigit():
        return word
    if length == 1:
            return word
    if punctuation_marks.__contains__(word[length - 1]):
        pos = length - 1
        if pos == 1:
            return word[0:pos]
        while punctuation_marks.__contains__(word[pos - 1]) and pos > 0:
            pos = pos - 1
        word = word[0:pos]
        length = len(word)
    try:
        if punctuation_marks.__contains__(word[0]):
            pos = 0
            while punctuation_marks.__contains__(word[pos + 1]):
                pos = pos + 1
            pos = pos + 1
            word = word[pos:length]
    except:
        return word
    return word

def measure_cos(vec1, vec2):
    n = len(vec1)
    denominator1, denominator2, numerator = 0, 0, 0
    for i in range(n):
        numerator += vec1[i] * vec2[i]
        denominator1 += vec1[i] * vec1[i]
        denominator2 += vec2[i] * vec2[i]
    return numerator / (sqrt(denominator1) * sqrt(denominator2))
