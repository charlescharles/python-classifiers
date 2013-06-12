import numpy as np

class NaiveBayesClassifier:
    def __init__(self, training_docs, training_classes):
        self.vocabulary = []
        self.classes = []
        self.priors = []
        self.conditionals = []
        self.train(training_docs, training_classes)

    def train(self, training_docs, training_classes):
        "Populate vocabulary and list of classes,      "
        " estimate prior and conditional probabilities."
        X, y = [], []
        for doc, cl in zip(training_docs, training_classes):
            if cl not in self.classes:
                self.classes.append(cl)
            vocab_vec, class_index = self._vectorize(doc), self.classes.index(cl)
            X.append(vocab_vec)
            y.append(class_index)

        # pad previous document word counts with zeros
        # TODO: numpy-ize this
        dimensions = max(map(len, X))
        for vec in X:
            if len(vec) < dimensions:
                vec.extend([0]*(dimensions-len(vec)))
        X = np.array(X)
        y = np.array(y)
        self._calculate()
        self.priors = np.array(self.priors)
        self.conditionals = np.array(self.conditionals)

    def _to_word(self, index):
        "Return the word represented by this index."
        return self.vocabulary[int(index)]

    def _to_class(self, index):
        "Return the class represented by this index."
        return self.classes[int(index)]

    def _w2i(self, word):
        "Return the index of this word in self.vocabulary."
        return self.vocabulary.index(word)

    def _c2i(self, classification):
        "Return the index of this class in self.classes."
        return self.classes.index(classification)

    def _vectorize(self, document):
        "Return a numerical vector representation of a collection of words."
        vocab_vector = list(np.zeros(len(self.vocabulary)))
        for word in document:
            if word not in self.vocabulary:
                self.vocabulary.append(word)
                vocab_vector.append(1)
            else:
                vocab_vector[self.vocabulary.index(word)] += 1
        return vocab_vector

    def _calculate(self):
        "Estimate prior probabilities of classes, P(class) and "
        "  conditional probabilities of words, P(word | class),"
        "  and append to self.priors and self.conditionals."
        for i in range(len(self.classes)):
            self.priors.append(float(np.sum(self.y==i))/np.sum(self.y))
            in_class = self.X[self.y==i]
            self.conditionals.append(np.sum(in_class, axis=0)/np.sum(in_class))

    def classify(self, document):
        "Return the class with the highest posterior probability."
        doc = self._vectorize(document)
        word_cond = np.sum(self.conditionals*doc, axis=1)
        log_prob = np.log(word_cond) + np.log(self.priors)
        cmax = np.argmax(log_prob)
        return self._to_class(cmax)


if __name__ == '__main__':
    doc1 = ["dog", "cat", "zebra", "kangaroo", "dog", "dog", "orange"]
    doc2 = ["orange", "apple", "starfruit", "grapefruit"]
    training_docs = [doc1, doc2]
    training_classes = ["animals", "fruits"]
    nb = NaiveBayesClassifier(training_docs, training_classes)
    print nb.classify(["orange", "apple", "dog"])