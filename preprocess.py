from nltk.corpus import brown
import numpy as np

class preprocess:
    """ Class for loading Brown files and preprocessing them """

    def __init__(self):
        self.fileids = brown.fileids()
        self.max_id = len(self.fileids)

    def load_a_file(self, fileid=None):
        if fileid is None or fileid>self.max_id:
            fileid = 0
        text = " ".join(brown.words(brown.fileids()[fileid]))
        text = text.lower()
        replacers = set([])
        for letter in text:
            if not (letter.isalpha() or letter == " "):
                replacers.add(letter)
        for letter in replacers:
            text = text.replace(" "+letter, "")
            text = text.replace(letter, "")
        return text

    def create_ngram_dic(self, n, file, dic={}):
        ngram_dic = dic
        for id_ in range(len(file)-n+1):
            key = file[id_:id_+n]
            if key in ngram_dic:
                ngram_dic[key] += 1
            else:
                ngram_dic[key] = 1
        return ngram_dic


def saveocd(ocd):
    with open("..\\Project\\test.txt", 'w') as file:
        for key in ocd:
            file.write(str(key) + "\t" + str(ocd[key]) + "\n")
        file.close()


if __name__ == "__main__":
    Pre = preprocess()
    totalchar = 0
    last_file = ""
    occurencemax = 12
    ocd = [{} for i in range(occurencemax)]
    for i in range(Pre.max_id):
        file_ = Pre.load_a_file(i)
        for o in range(len(ocd)):  # Update occurence-counter dictionaries
            ocd[o] = Pre.create_ngram_dic(o+1, file_, ocd[o])
        print(i)
    for o in ocd:
        print(len(o))
    val_list = []
    for key in ocd[1]:
        val_list.append(ocd[1][key])
    # saveocd(ocd[-1])
    std = np.std(val_list)
    print(val_list/std)
    val_list = []
    for key in ocd[-1]:
        val_list.append(ocd[-1][key])
    # saveocd(ocd[-1])
    std = np.std(val_list)
    print(val_list[43:46] / std)
