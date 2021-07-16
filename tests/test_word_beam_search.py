import codecs

import numpy as np
from word_beam_search import WordBeamSearch


def apply_word_beam_search(mat, corpus, chars, word_chars):
    """Decode using word beam search. Result is tuple, first entry is label string, second entry is char string."""
    T, B, C = mat.shape

    # decode using the "Words" mode of word beam search with beam width set to 25 and add-k smoothing to 0.0
    assert len(chars) + 1 == C

    wbs = WordBeamSearch(25, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), word_chars.encode('utf8'))
    label_str = wbs.compute(mat)

    # result is string of labels terminated by blank
    char_str = []
    for curr_label_str in label_str:
        s = ''
        for label in curr_label_str:
            s += chars[label]  # map label to char
        char_str.append(s)
    return label_str[0], char_str[0]


def load_mat(fn):
    """Load matrix from csv and apply softmax."""

    mat = np.genfromtxt(fn, delimiter=';')[:, :-1]  # load matrix from file
    T = mat.shape[0]  # dim0=t, dim1=c

    # apply softmax
    res = np.zeros(mat.shape)
    for t in range(T):
        y = mat[t, :]
        e = np.exp(y)
        s = np.sum(e)
        res[t, :] = e / s

    # expand to TxBxC
    return np.expand_dims(res, 1)


def test_mini_example():
    """Mini example, just to check that everything is working."""
    corpus = 'a ba'  # two words "a" and "ba", separated by whitespace
    chars = 'ab '  # the first three characters which occur in the matrix (in this ordering)
    word_chars = 'ab'  # whitespace not included which serves as word-separating character
    mat = np.array([[[0.9, 0.1, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]],
                    [[0.6, 0.4, 0.0, 0.0]]])  # 3 time-steps and 4 characters per time time ("a", "b", " ", blank)

    res = apply_word_beam_search(mat, corpus, chars, word_chars)
    print('')
    print('Mini example:')
    print('Label string:', res[0])
    print('Char string:', '"' + res[1] + '"')
    assert res[1] == 'ba'


def test_real_example():
    """Real example using a sample from a HTR dataset."""
    data_path = '../data/bentham/'
    corpus = codecs.open(data_path + 'corpus.txt', 'r', 'utf8').read()
    chars = codecs.open(data_path + 'chars.txt', 'r', 'utf8').read()
    word_chars = codecs.open(data_path + 'wordChars.txt', 'r', 'utf8').read()
    mat = load_mat(data_path + 'mat_2.csv')

    res = apply_word_beam_search(mat, corpus, chars, word_chars)
    print('')
    print('Real example:')
    print('Label string:', res[0])
    print('Char string:', '"' + res[1] + '"')
    assert res[1] == 'submitt both mental and corporeal, is far beyond any idea'
