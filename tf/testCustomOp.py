from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf
import codecs


def testCustomOp(feedMat, corpus, chars, wordChars):
	"decode using word beam search. Result is tuple, first entry is label string, second entry is char string."

	# TF session
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# load custom TF op
	word_beam_search_module = tf.load_op_library('../cpp/proj/TFWordBeamSearch.so')

	# input with shape TxBxC
	mat=tf.placeholder(tf.float32, shape=feedMat.shape)

	# decode using the "Words" mode of word beam search with beam width set to 25 and add-k smoothing to 0.0
	assert len(chars) + 1 == mat.shape[2]
	decode = word_beam_search_module.word_beam_search(mat, 25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))

	# feed matrix of shape TxBxC and evaluate TF graph
	res = sess.run(decode, { mat:feedMat })

	# result is string of labels terminated by blank (similar to C-strings) if shorter than T
	blank = len(chars)
	s = ''
	for label in res[0]:
		if label == blank:
			break
		s += chars[label] # map label to char
	return (res[0], s)


def loadMat(fn):
	"load matrix from csv and apply softmax"

	mat = np.genfromtxt(fn, delimiter=';')[:,:-1] #load matrix from file
	maxT, _ = mat.shape # dim0=t, dim1=c
	
	# apply softmax
	res = np.zeros(mat.shape)
	for t in range(maxT):
		y = mat[t, :]
		e = np.exp(y)
		s = np.sum(e)
		res[t, :] = e / s

	# expand to TxBxC
	return np.expand_dims(res, 1)


def testMiniExample():
	"mini example, just to check that everything is working"
	corpus = 'a ba' # two words "a" and "ba", separated by whitespace
	chars = 'ab ' # the first three characters which occur in the matrix (in this ordering)
	wordChars = 'ab' # whitespace not included which serves as word-separating character
	mat = np.array([[[0.9, 0.1, 0.0, 0.0]],[[0.0, 0.0, 0.0, 1.0]],[[0.6, 0.4, 0.0, 0.0]]]) # 3 time-steps and 4 characters per time time ("a", "b", " ", blank)

	res = testCustomOp(mat, corpus, chars, wordChars)	
	print('')
	print('Mini example:')
	print('Label string:', res[0])
	print('Char string:', '"' + res[1] + '"')
	assert res[1] == 'ba'


def testRealExample():
	"real example using a sample from a HTR dataset"
	dataPath = '../data/bentham/'
	corpus = codecs.open(dataPath + 'corpus.txt', 'r', 'utf8').read()
	chars = codecs.open(dataPath + 'chars.txt', 'r', 'utf8').read()
	wordChars = codecs.open(dataPath + 'wordChars.txt', 'r', 'utf8').read()
	mat = loadMat(dataPath + 'mat_2.csv')

	res = testCustomOp(mat, corpus, chars, wordChars)
	print('')
	print('Real example:')
	print('Label string:', res[0])
	print('Char string:', '"' + res[1] + '"')
	assert res[1] == 'submitt both mental and corporeal, is far beyond any idea'


if __name__=='__main__':
	# test custom op
	testMiniExample()
	testRealExample()

