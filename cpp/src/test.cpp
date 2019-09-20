#include "test.hpp"
#include "LanguageModel.hpp"
#include "PrefixTree.hpp"
#include "MatrixCSV.hpp"
#include "Metrics.hpp"
#include "WordBeamSearch.hpp"
#include "DataLoader.hpp"
#include <cassert>
#include <iostream>


// tests for the classes, run in debug mode (assert)
void test()
{
#ifdef NDEBUG
	std::cout << "UNITTESTS: must run in debug mode";
	return;
#endif

	std::cout << "UNITTESTS: begin\n";


	// test language model
	LanguageModel lm("this is a text. this and that.", "abcdefghijklmnopqrstuvwxyz., ","abcdefghijklmnopqrstuvwxyz", LanguageModelType::NGrams);
	assert(lm.getUnigramProb(lm.utf8ToLabel("this")) == 2.0 / 7.0);
	assert(lm.getUnigramProb(lm.utf8ToLabel("yyy")) == 0.0);
	assert(lm.getBigramProb(lm.utf8ToLabel("this"), lm.utf8ToLabel("and")) == 1.0 / 2.0);
	assert(lm.getBigramProb(lm.utf8ToLabel("this"), lm.utf8ToLabel("that")) == 0.0);
	assert(lm.getBigramProb(lm.utf8ToLabel("this"), lm.utf8ToLabel("yyy")) == 0.0);
	

	// test prefix tree, use language model to map between utf8 and label strings
	PrefixTree t;
	t.addWord(lm.utf8ToLabel("this"));
	t.addWord(lm.utf8ToLabel("that"));
	t.allWordsAdded();
	assert(lm.labelToUtf8(t.getNextChars(lm.utf8ToLabel("th"))) == "ai");
	assert(lm.labelToUtf8(t.getNextWords(lm.utf8ToLabel("thi"))[0]) == "this");
	assert(lm.labelToUtf8(t.getNextWords(lm.utf8ToLabel("that"))[0]) == "that");
	assert(t.isWord(lm.utf8ToLabel("that")) == true);
	assert(t.isWord(lm.utf8ToLabel("yyy")) == false);


	// test matrix class by reading from a csv file
	MatrixCSV mat("../../data/iam/mat_0.csv");
	assert(mat.rows() == 100);
	assert(mat.cols() == 80);
	assert(mat.getAt(0, 0) == 0.946499);
	assert(mat.getAt(mat.rows()-1, mat.cols()-1) == 8.68117);


	// metrics (CER/WER)
	Metrics metrics{ lm.getWordChars() };
	metrics.addResult({}, {});
	assert(metrics.getCER() == 0.0);
	assert(metrics.getWER() == 0.0);
	metrics.addResult(lm.utf8ToLabel("hello"), lm.utf8ToLabel("hxello"));
	assert(metrics.getCER() == 1.0/5.0);
	assert(metrics.getWER() == 1.0);
	metrics.addResult(lm.utf8ToLabel("hello world "), lm.utf8ToLabel("hello wxrld "));
	assert(metrics.getCER() == 2.0 / 17.0);
	assert(metrics.getWER() == 2.0/3.0);


	// decode
	DataLoader loader("../../data/test/", 1, LanguageModelType::NGrams);
	const auto data=loader.getNext();
	const auto decoded=wordBeamSearch(data.mat, 10, loader.getLanguageModel(), LanguageModelType::Words);
	assert(loader.getLanguageModel()->labelToUtf8(decoded) == "ba");

	
	std::cout << "UNITTESTS: end\n";
}