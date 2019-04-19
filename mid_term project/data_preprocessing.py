from konlpy.tag import Kkma


def generate_corpus(data):
    kkma = Kkma()

    sentences = []
    list_vec = []
    word_list = []

    for k in data:
        sentences.append(kkma.sentences(k[0]))
        for sentence in sentences:
            for word in sentence:
                list_vec.append(kkma.nouns(word))

    for l in list_vec:
        empty_vec = []
        for word in l:
            if len(word) >= 2:
                empty_vec.append(word)
        word_list.append(empty_vec)

    return word_list
