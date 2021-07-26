#pragma once
#include "HashFunction.hpp"
#include "PrefixTree.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <stdint.h>
#include <cstddef>


// scoring mode of LM
enum class LanguageModelType
{
	Words // use no N-grams, but restrict output to words from corpus (very fast)
	, NGrams // consider N-grams each time when beam text finishes a new word (fast)
	, NGramsForecast // consider N-grams for possible following words each time a characters is added to beam text (very slow)
	, NGramsForecastAndSample // consider N-grams for subset of possible following words each time a characters is added to beam text (slow)
};


// unigram and bigram LM with add-k smoothing
class LanguageModel
{
public:
	// CTOR
	LanguageModel(const std::string& corpus, const std::string& chars, const std::string& wordChars, LanguageModelType lmType, double addK = 0.0);

	// unigram and bigram probability
	double getUnigramProb(const std::vector<uint32_t>& w) const;
	double getBigramProb(const std::vector<uint32_t>& w1, const std::vector<uint32_t>& w2) const;

	// given some text, check if it is a word, give next possible words, give next possible characters
	bool isWord(const std::vector<uint32_t>& text) const;
	std::vector<std::vector<uint32_t>> getNextWords(const std::vector<uint32_t>& text) const;
	std::vector<uint32_t> getNextChars(const std::vector<uint32_t>& text) const;

	// char sets
	const std::set<uint32_t>& getAllChars() const; 
	const std::set<uint32_t>& getWordChars() const; 
	const std::set<uint32_t>& getNonWordChars() const;

	// utf8 -> label ->utf8
	std::vector<uint32_t> utf8ToLabel(const std::string& utf8Str); 
	std::string labelToUtf8(const std::vector<uint32_t>& labelStr); 

private:
	// words, unigrams and bigrams
	struct Bigram
	{
		size_t count = 0;
		double prob = 0.0;
	};

	struct Unigram
	{
		size_t count = 0;
		double prob = 0.0;
		std::unordered_map<std::vector<uint32_t>, Bigram, HashFunction> bigrams;
	};

	std::unordered_map<std::vector<uint32_t>, Unigram, HashFunction> m_unigrams;

	double m_addK = 0.0; // add-k smoothing

	// prefix tree
	PrefixTree m_tree;

	// map between label strings, utf8 strings and unicode strings
	std::vector<uint32_t> m_labelToCodepoint; // label->unicode
	std::unordered_map<uint32_t, uint32_t> m_codepointToLabel; // unicode->label

	// sets of labels
	void initLabelSets(const std::unordered_map<uint32_t, uint32_t>& codepointToLabelMapping, const std::vector<uint32_t>& wordCodepoints);
	std::set<uint32_t> m_allLabels;
	std::set<uint32_t> m_wordLabels;
	std::set<uint32_t> m_nonWordLabels;

	// map between utf8, codepoints and labels
	std::vector<uint32_t> utf8ToCodepoint(const std::string& s);
	std::unordered_map<uint32_t, uint32_t> codepointToLabelMapping(const std::vector<uint32_t>& charsCP);
	
};

