from nltk.corpus import brown
import nltk
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split as tts
from operator import itemgetter
import pickle


class PreProcess:
    """ Class for loading Brown files and pre-processing them """

    def __init__(self, max_ocd_size=7, do_count=True):
        self.file_ids = brown.fileids()
        self.max_id = len(self.file_ids)
        self.max_ocd_size = max_ocd_size
        self.all_files = []
        self.fgram_files = []
        if do_count is True:
            self.ocds = self.count_occurrences()

    def count_occurrences(self, file_int_ids=None,
                          file_cat_ids=None, testing=False):
        print("Counting k-gram occurrences in files")
        ocds = [{} for _ in range(self.max_ocd_size)]
        if file_cat_ids is None:
            if file_int_ids is None:
                file_int_ids = range(self.max_id)
            for i in tqdm(file_int_ids):
                file_ = self.load_a_file(fileid_int=i)
                self.all_files.append(file_)
                # Update occurrence-counter dictionaries
                if testing is False:
                    for j in range(self.max_ocd_size):
                        ocds[j] = self.create_ngram_dic(j + 1, file_, ocds[j])
        else:
            for i in tqdm(file_cat_ids):
                file_ = self.load_a_file(fileid_cat=i)
                self.all_files.append(file_)
                if testing is False:
                    for j in range(self.max_ocd_size):
                        ocds[j] = self.create_ngram_dic(j + 1, file_, ocds[j])
        return ocds

    @staticmethod
    def calc_std_of_ocd(ocd):
        val_list = []
        for key in ocd:
            val_list.append(ocd[key])
        std = np.std(val_list)
        return std

    def load_a_file(self, fileid_int=None, fileid_cat=None):
        """

        :int fileid_int: id of the brown file that will be loaded
        """
        if fileid_cat is None:
            if fileid_int is None or fileid_int > self.max_id:
                fileid_int = 0
            text = " ".join(brown.words(brown.fileids()[fileid_int]))
        else:
            text = " ".join(brown.words(fileid_cat))
        text = text.lower()
        replacers = set([])
        for letter in text:
            if not (letter.isalpha() or letter == " "):
                replacers.add(letter)
        for letter in replacers:
            text = text.replace(" "+letter, "")
            text = text.replace(letter, "")
        return text

    @staticmethod
    def create_ngram_dic(n, file, dic=None):
        if dic is None:
            dic = {}
        ngram_dic = dic
        for id_ in range(len(file)-n+1):
            ocd_key = file[id_:id_+n]
            if ocd_key in ngram_dic:
                ngram_dic[ocd_key] += 1
            else:
                ngram_dic[ocd_key] = 1
        return ngram_dic

    def get_norm_freq_occurrences(self, cat_ids, testing):
        freq_ocds = self.count_occurrences(file_cat_ids=cat_ids,
                                           testing=testing)
        if testing is True:
            return None
        norm_freq_ocds = []
        print("normalizing results")
        for freq_ocd in tqdm(freq_ocds):
            nfo = self.normalize_occurences(freq_ocd)
            norm_freq_ocds.append(nfo)
        return norm_freq_ocds

    def normalize_occurences(self, ocd):
        norm_std = self.calc_std_of_ocd(ocd)
        norm_ocd = ocd
        for key in norm_ocd:
            norm_ocd[key] = norm_ocd[key] / norm_std
        return norm_ocd

    def make_fgrams(self, ocd, allfiles=True, fileids=None, testing=False):
        print("making f-grams")
        if allfiles is True:
            i = 0
            if testing is True:
                i = 1000
            for file_ in tqdm(self.all_files):
                i += 1
                try:
                    fname = "./data/fgram_file" + str(i)+".pkl"
                    fgram_file = pickle.load(open(fname, 'rb'))
                except:
                    print("failed to load file")
                    revtext = file_[:]
                    fgram_list = []
                    for kgram in ocd:
                        substr = kgram[0]
                        repl_text = "_" * len(substr)
                        while substr in revtext:
                            loc = revtext.find(substr)
                            revtext = revtext.replace(substr, repl_text, 1)
                            fgram_list.append([substr, loc])
                    fgram_list = sorted(fgram_list, key=itemgetter(1))
                    fgram_file = [revtext[0] for revtext in fgram_list]
                    pickle.dump(fgram_file,
                                open("./data/fgram_file"+str(i)+".pkl", 'wb'))
                self.fgram_files.append(fgram_file)


class NaiveBayesKGrams:
    def __init__(self, k, ocd, traindata, testdata, priors, maybemore=None):
        print("testing for {0}-grams.".format(k))
        self.k = k
        self.ocd = ocd
        self.trainfiles, self.traincats = traindata
        self.testfiles, self.testcats = testdata
        self.trainkgramfiles = self.makekgram(self.trainfiles)
        self.size_voc = self.calc_size_voc(self.trainfiles)
        self.rw_cat = self.count_running_words()
        self.testkgramfiles = self.makekgram(self.testfiles, True)
        self.priors = priors
        self.occ_cat = self.count_occ_in_cat()
        self.test_score_data()

    def test_score_data(self):
        tot_correct = 0
        for id_, file_ in enumerate(tqdm(self.testkgramfiles)):
            if self.calc_cat_prob_for_test_file(id_) == testcat[id_]:
                tot_correct += 1
        print("From the {0} test files, {1} were classified correctly".format(len(self.testfiles), tot_correct))

    def calc_size_voc(self, files):
        voc = set([])
        for file_ in files:
            for kgram in file_:
                voc.add(str(kgram))
        return len(voc)

    def count_running_words(self):
        running_words_per_cat = {}
        for id_, cat in enumerate(self.traincats):
            ca = cat[1]
            if ca in running_words_per_cat:
                running_words_per_cat[ca] += len(self.trainkgramfiles)
            else:
                running_words_per_cat[ca] = len(self.trainkgramfiles)
        return running_words_per_cat

    def makekgram(self, files, files_are_for_testing=False):
        fname = "./data/" + str(self.k) + "gramfile"
        kgramfiles = []
        i = 0
        if files_are_for_testing:
            i = 1000
        for file_ in tqdm(files):
            i += 1
            kgramfname = fname + str(i) + ".pkl"
            try:
                kgramfile = pickle.load(open(kgramfname, 'rb'))
            except:
                kgramfile = []
                for idx in range(len(file_)-self.k):
                    kgram = file_[idx:idx+self.k]
                    kgramfile.append(kgram)
                pickle.dump(kgramfile, open(kgramfname, 'wb'))
            kgramfiles.append(kgramfile)
        return kgramfiles

    def calc_most_prob_cat(self, file_id):
        probs = self.calc_cat_prob_for_file(file_id)
        e = None
        p = None
        winning_cat = None
        for cat in probs:
            if e is None:
                p, e = probs[cat]
                winning_cat = cat
            elif probs[cat][1] == e:
                if probs[cat][0] > p:
                    winning_cat = cat
                    p, e = probs[cat]
            else:
                if probs[cat][1] > e:
                    winning_cat = cat
                    p, e = probs[cat]
        return winning_cat, p, e

    def calc_cat_prob_for_test_file(self, file_id):
        probs = {}
        for cat in self.rw_cat:
            cat_prob = 1
            e = 0
            for fgram in self.trainkgramfiles[file_id]:
                term_prob = (self.occ_in_cat(fgram, cat)) / \
                            (self.rw_cat[cat] + self.size_voc)
                # times hundred in order to not have a too low value for p.
                cat_prob = cat_prob * term_prob
                while cat_prob < 0.1:
                    cat_prob *= 10
                    e -= 1
            probs[cat] = (cat_prob, e)
        return probs

    def calc_cat_prob_for_train_file(self, file_id):
        probs = {}
        for cat in self.rw_cat:
            cat_prob = 1
            e = 0
            for kgram in self.testkgramfiles[file_id]:
                term_prob = (self.occ_in_cat(str(kgram), str(cat))) / \
                            (self.rw_cat[cat] + self.size_voc)
                # times hundred in order to not have a too low value for p.
                cat_prob = cat_prob * term_prob
                while cat_prob < 0.1:
                    cat_prob *= 10
                    e -= 1
            probs[cat] = (cat_prob, e)
        return probs

    def occ_in_cat(self, kgram, cat):
        """ return occurence in category. Added one for laplace smoothing"""
        if kgram in self.occ_cat[cat]:
            return self.occ_cat[cat][kgram] + 1
        return 1

    def count_occ_in_cat(self):
        occ_words_per_cat = {}
        for id_, cat in enumerate(self.traincats):
            ca = cat[1]
            if ca not in occ_words_per_cat:
                occ_words_per_cat[ca] = {}
            for lkgram in self.trainkgramfiles:
                kgram = str(lkgram)
                if kgram in occ_words_per_cat[ca]:
                    occ_words_per_cat[ca][kgram] += 1
                else:
                    occ_words_per_cat[ca][kgram] = 1
        return occ_words_per_cat



class NaiveBayes:
    """Class that handles the classifying of the naive bayes"""
    def __init__(self, cat_ids, prepro, testing=False, trainnb=None):
        """
        :list of strings cat_ids: list of the file-ids that will be used
            to train the data. Ordered by category type.
        :PreProcessing prepro: the class to preprocess the files
        """
        self.cat_ids = cat_ids
        self.priors = self.calc_priors()
        self.cats = self.find_cats()
        self.prepro = prepro
        self.prepro.all_files = []
        self.prepro.fgram_files = []
        self.norm_freq_ocds = self.get_ocds(self.all_ids(), testing)
        if testing is False:
            self.sorted_combined_ocd = self.sort_ocds()
            self.prepro.make_fgrams(self.sorted_combined_ocd, testing=testing)
            print("check pass")
            self.rw_cat = self.count_running_words()
            self.size_voc = self.count_vocabulary()
            self.occ_cat = self.count_occ_in_cat()
        else:
            self.prepro.make_fgrams(trainnb.sorted_combined_ocd, testing=testing)
            self.rw_cat = trainnb.rw_cat
            self.size_voc = trainnb.size_voc
            self.occ_cat = trainnb.occ_cat

    def find_cats(self):
        cats = []
        for cat in self.priors:
            cats.append(cat)
        return cats

    def calc_priors(self):
        """ Calculates the priors for all classes. """
        priordic = {}
        total = 0
        for cat in self.cat_ids:
            category = cat[0][1]
            priordic[category] = len(cat)
            total += len(cat)
        for category in priordic:
            priordic[category] = priordic[category] / float(total)
        return priordic

    def sort_ocds(self):
        big_ocd = {}
        print("sorting")
        for ocd in tqdm(self.norm_freq_ocds):
            big_ocd.update(ocd)
        return self.sort_dic_by_item(big_ocd)

    def get_ocds(self, cat_ids, testing):
        ocds = self.prepro.get_norm_freq_occurrences(cat_ids, testing)
        return ocds

    @staticmethod
    def sort_dic_by_item(ocd):
        return sorted(ocd.items(), key=itemgetter(1), reverse=True)

    def all_ids(self):
        all_cat_ids = []
        for cat in self.cat_ids:
            for id_ in cat:
                all_cat_ids.append(id_)
        return all_cat_ids

    def print_ocds(self, printlen=True):
        print("priors:")
        for ocd in self.norm_freq_ocds:
            if printlen is True:
                print(len(ocd))
            else:
                print(ocd)

    def count_occ_in_cat(self):
        ids = self.all_ids()
        occ_words_per_cat = {}
        for id_, cat in enumerate(ids):
            ca = cat[1]
            if ca not in occ_words_per_cat:
                occ_words_per_cat[ca] = {}
            for fgram in self.prepro.fgram_files[id_]:
                if fgram in occ_words_per_cat[ca]:
                    occ_words_per_cat[ca][fgram] += 1
                else:
                    occ_words_per_cat[ca][fgram] = 1
        return occ_words_per_cat

    def occ_in_cat(self, fgram, cat):
        """ return occurence in category. Added one for laplace smoothing"""
        if fgram in self.occ_cat[cat]:
            return self.occ_cat[cat][fgram]+1
        return 1

    def count_running_words(self):
        ids = self.all_ids()
        running_words_per_cat = {}
        for id_, cat in enumerate(ids):
            ca = cat[1]
            if ca in running_words_per_cat:
                running_words_per_cat[ca] += len(self.prepro.fgram_files[id_])
            else:
                running_words_per_cat[ca] = len(self.prepro.fgram_files[id_])
        return running_words_per_cat

    def count_vocabulary(self):
        voc = set([])
        for file_ in tqdm(self.prepro.fgram_files):
            for fgram in file_:
                voc.add(fgram)
        # ###TESTING
        # for v in voc:
        #     if len(v)==11:
        #         print(sorted(list(v)))
        # ###/TESTING
        return len(voc)

    def calc_cat_prob_for_file(self, file_id):
        probs = {}
        for cat in self.rw_cat:
            cat_prob = 1
            e = 0
            for fgram in self.prepro.fgram_files[file_id]:
                term_prob = (self.occ_in_cat(fgram, cat)) /\
                             (self.rw_cat[cat]+self.size_voc)
                # times hundred in order to not have a too low value for p.
                cat_prob = cat_prob * term_prob
                while cat_prob < 0.1:
                    cat_prob *= 10
                    e -= 1
            probs[cat] = (cat_prob, e)
        return probs

    def calc_most_probable_cat(self, file_id, testing=False):
        probs = self.calc_cat_prob_for_file(file_id)
        e = None
        p = None
        winning_cat = None
        for cat in probs:
            if e is None:
                p, e = probs[cat]
                winning_cat = cat
            elif probs[cat][1] == e:
                if probs[cat][0] > p:
                    winning_cat = cat
                    p, e = probs[cat]
            else:
                if probs[cat][1]>e:
                    winning_cat = cat
                    p, e = probs[cat]
        ###TESTING
        if testing:
            print(probs)
        ###/TESTING
        return winning_cat, p, e

def create_train_test_ids():
    train_id = []
    test_id = []
    brown_categories = brown.categories()
    for category in brown_categories:
        ttid = tts(brown.fileids(category))
        train_id.append(ttid[0])
        test_id.append(ttid[1])
    pickle.dump((train_id, test_id), open("./data/train_test_ids.pkl", 'wb'))
    return train_id, test_id


def load_train_test_ids(pfile="./data/train_test_ids.pkl"):
    (train_ids, test_ids) = pickle.load(open(pfile, 'rb'))
    return train_ids, test_ids


def save_ocd(ocd):
    with open("..\\Project\\test.txt", 'w') as file:
        for ocd_key in ocd:
            file.write(str(ocd_key) + "\t" + str(ocd[ocd_key]) + "\n")
        file.close()


def set_start_values():
    occurrence_max = 11  # Memoryerror at 12
    counted = False  # Will create ocds for all files (test and train)
    first_time = False  # Only very first time to split test and train
    return occurrence_max, counted, first_time


def testNB(pre, nb, testing=False):
    tot_correct = 0
    tot_files = len(pre.fgram_files)
    print("testing correctness")
    for file_id in tqdm(range(tot_files)):
        wc, p, e = nb.calc_most_probable_cat(file_id)
        if wc == nb.all_ids()[file_id][1]:
            tot_correct += 1
    print("from {0} files, {1} were guessed correctly".format(tot_files,
                                                              tot_correct))


def testk(k, p, n, trainfiles, testfiles):
    nbk = NaiveBayesKGrams(k, p[k - 1], trainfiles, testfiles, n.priors)


if __name__ == "__main__":
    # nltk.download()
    occ_max, count, first = set_start_values()
    if first:
        traincat, testcat = create_train_test_ids()
    else:
        traincat, testcat = load_train_test_ids()
    pre = PreProcess(occ_max, count)
    nb = NaiveBayes(traincat, pre)
    trainfiles = pre.all_files[:]
    ocds = nb.norm_freq_ocds[:]
    nb2 = NaiveBayes(testcat, pre, testing=True, trainnb=nb)
    testfiles = pre.all_files[:]
    testNB(pre, nb, True)
    for k in range(1,11):
        testk(k, ocds, nb, (trainfiles, traincat), (testfiles, testcat))
