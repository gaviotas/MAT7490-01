from konlpy.tag import Mecab


def generate_corpus(data):
    mecab = Mecab()

    word_list = []

    for i, k in enumerate(data):
        tokens = mecab.nouns(k)
        tokens = [n for n in tokens if len(n) >= 2]
        word_list.append(tokens)
        # if i % 1000 == 0:
        #     print(i)
    return word_list
