# coding: utf-8

import numpy as np
from numpy.linalg import norm
from collections import defaultdict





class word2vec():

    def __init__(self, lr=1, epochs=1000, window_size=2, learning_rate = 0.1):
        self.lr = lr
        self.epochs = epochs
        self.window_size = window_size
        self.learning_rate = learning_rate

    def preprocessing(self, inputs):
        tokens = inputs.strip().split()
        word_to_idx = dict([[word, idx] for idx, word in enumerate(set(tokens))])
        idx_to_word = dict([[idx, word] for idx, word in enumerate(set(tokens))])
        num_words = len(set(tokens))

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.num_words = num_words

    def generating_dataset(self, corpus):

        self.embedding = self.initialize_W(self.num_words, 20)
        self.W1 = self.initialize_W(20, self.num_words)
        self.W2 = self.initialize_W(self.num_words, self.num_words)

        dataset = []
        # for sentence in corpus:
        sentence = corpus.strip().split()
        # print(sentence)
        for center_idx, center_word in enumerate(sentence):
            for window in range(max(center_idx - self.window_size,0), min(center_idx + self.window_size + 1, self.num_words)):
                if window == center_idx:
                    continue
                dataset.append([self.word_to_idx[center_word], self.word_to_idx[sentence[window]]])
        # return np.array(dataset).reshape(1, -1)\
        return np.array(dataset)


    def check_dataset(self, dataset):

        for i in range(dataset.shape[0]):
            print(self.idx_to_word[dataset[i][0]], self.idx_to_word[dataset[i][1]])

    def onehot(self,input,length):
        onehot_vector = np.zeros(length).reshape(length)
        onehot_vector[input] = 1
        return onehot_vector

    def train(self, dataset, batch_size=32):

        x_train = dataset[:,0].reshape(1,-1)
        y_train = dataset[:,1].reshape(1,-1)

        for epoch in range(self.epochs):
            self.loss = 0
            batch_idx = list(range(0, x_train.shape[1], batch_size))
            for j in batch_idx:
                x_batch = x_train[:, j:j+batch_size]
                y_batch = y_train[:, j:j+batch_size]
                out, cache = self.forward(x_batch)
                loss, dout = self.softmax_loss(out, y_batch)
                dout, dword_vector, dW1 = self.backward(dout, cache)
                self.optimize(self.learning_rate, x_batch, dword_vector, dW1)
            if (epoch % 10) == 0:
                print('Epoch: [{0}/{1}]\t' 'Loss: {2}'.format(epoch, self.epochs, loss))

    def forward(self,x):
        word_vector, cache1 = self.get_word2vec(x)
        out, cache2 = self.affine_forward(word_vector, self.W1)
        cache = cache1, cache2
        return out, cache

    def get_word2vec(self, x):
        length = x.shape[1]
        word_vector = self.embedding[x.flatten(), :]
        cache = x, length
        return word_vector, cache

    def affine_forward(self, x, W):
        out = x.dot(W)
        cache = x, W
        return out, cache

    def backward(self, dout, cache):
        cache1, cache2 = cache
        dword_vector, dW1 = self.affine_backward(dout, cache2)
        # dinput = self.get_word2vec_backward(dword_vector, cache1)
        return dout, dword_vector, dW1

    def get_word2vec_backward(self, dout, cache):
        pass

    def affine_backward(self, dout, cache):
        x, W = cache
        dx, dW = None, None

        dx = np.dot(dout,W.T).reshape(x.shape)
        dW = np.dot(x.reshape(x.shape[0], -1).T, dout)

        return dx, dW


    def optimize(self, learning_rate, x, dword_vector, dW1):
        self.embedding[x.flatten(), :] -= learning_rate * dword_vector
        self.W1 -= learning_rate * dW1


    def initialize_W(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * 0.1
    def softmax_loss(self, x, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
          class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
          0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = x.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx


    def return_word_vector(self, word):
        idx = self.word_to_idx[word]
        return self.embedding[idx]

    def return_similar_word(self, word):
        idx = self.word_to_idx[word]
        cur_embedding = self.embedding[idx]
        similarity_dict = {}
        for others in range(self.num_words):
            if others == idx:
                continue
            other_embedding = self.embedding[others]
            # similarity = np.dot(cur_embedding, other_embedding)
            # similarity /= np.linalg.norm(cur_embedding)
            # similarity /= np.linalg.norm(other_embedding)
            similarity = cos_similarity(cur_embedding, other_embedding)
            similarity_dict[others] = similarity
        top = sorted(similarity_dict.items(), key = lambda kv: kv[1], reverse=True)
        print("Order \t\t Word \t\t\t Similarity")
        for num, (word, sim) in enumerate(top):
            print("{0}\t\t{1}\t\t\t{2:.2f}".format(num+1, self.idx_to_word[word], sim))

def cos_similarity (input1,input2):
	a = np.array(input1)
	b = np.array(input2)
	result = np.inner(a,b) / (norm(a)*norm(b))
	return result

def main():
    input_data = 'natural language processing and machine learning is fun and exciting'

    words = word2vec()

    words.preprocessing(input_data)
    dataset = words.generating_dataset(input_data)
    words.train(dataset)

    word = 'processing'
    embedding = words.return_word_vector(word)
    print(embedding)
    words.return_similar_word(word)
if __name__ == '__main__':
    main()
