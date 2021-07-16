import editdistance

from DataLoader import DataLoader
from Metrics import Metrics
from WordBeamSearch import wordBeamSearch


def main():
    # Settings
    sampleEach = 1
    dataset = 'bentham'
    useNGrams = True

    # load dataset
    loader = DataLoader(dataset, sampleEach)
    print('Decoding ' + str(loader.getNumSamples()) + ' samples now.')
    print('')

    # metrics calculates CER and WER for dataset
    m = Metrics(loader.lm.getWordChars())

    # decode each sample from dataset
    for idx, data in enumerate(loader):
        # decode matrix
        res = wordBeamSearch(data.mat, 10, loader.lm, useNGrams)
        print('Sample: ' + str(idx + 1))
        print('Filenames: ' + data.fn)
        print('Result:       "' + res + '"')
        print('Ground Truth: "' + data.gt + '"')
        strEditDist = str(editdistance.eval(res, data.gt))
        print('Editdistance: ' + strEditDist)

        # output CER and WER
        m.addSample(data.gt, res)
        print('Accumulated CER and WER so far:', 'CER:', m.getCER(), 'WER:', m.getWER())
        print('')


if __name__ == '__main__':
    main()
