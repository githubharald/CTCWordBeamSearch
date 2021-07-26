#include "LanguageModel.hpp"
#include "utfcpp/utf8.h"
#include <set>
#include <iostream>
#include <algorithm>
#include <cassert>


LanguageModel::LanguageModel(const std::string& corpus, const std::string& chars, const std::string& wordChars, LanguageModelType lmType, double addK)
:m_addK(addK)
{
	m_labelToCodepoint=utf8ToCodepoint(chars);
	m_codepointToLabel = codepointToLabelMapping(m_labelToCodepoint);
	const auto wordCodepoints = utf8ToCodepoint(wordChars);
	initLabelSets(m_codepointToLabel, wordCodepoints);

	// extract words and put them in list
	auto iter = corpus.begin();
	auto end = corpus.end();
	std::vector<uint32_t> currWord;
	std::vector<std::vector<uint32_t>> wordList;
	while (iter != end)
	{
		uint32_t c = utf8::next(iter, end);
		const bool isWordChar = std::find(wordCodepoints.begin(), wordCodepoints.end(), c) != wordCodepoints.end();
		if (isWordChar)
		{
			currWord.push_back(m_codepointToLabel[c]);
		}
		if ((!isWordChar || iter==end) && !currWord.empty())
		{
			wordList.push_back(currWord);
			currWord.clear();
		}
	}


	// calc unigram, add words to tree
	const double wordWeight = 1.0 / wordList.size();
	for (const auto& w : wordList)
	{
		// not yet in list: add to list and add to tree
		if (m_unigrams.find(w) == m_unigrams.end())
		{
			// add to prefix tree
			m_tree.addWord(w);

			// assign unique id
			m_unigrams[w] = Unigram();
		}

		Unigram& unigram = m_unigrams[w];
		unigram.prob += wordWeight;
		unigram.count++;
	}

	// all words are added, reorganize tree for faster access
	m_tree.allWordsAdded();

	// leave CTOR if no NGrams are needed
	if (lmType == LanguageModelType::Words)
	{
		return;
	}

	// calc bigrams
	for (size_t i=0; !wordList.empty() && i<wordList.size()-1; ++i)
	{
		const std::vector<uint32_t>& w1 = wordList[i];
		const std::vector<uint32_t>& w2 = wordList[i+1];

		// insert new bigram if needed
		assert(m_unigrams.find(w1) != m_unigrams.end());
		if (m_unigrams[w1].bigrams.find(w2) == m_unigrams[w1].bigrams.end())
		{
			m_unigrams[w1].bigrams[w2] = Bigram();
		}

		// add 1 to bigram count
		Bigram& bigram = m_unigrams[w1].bigrams[w2];
		bigram.count++;
	}

	// normalize bigrams
	for (auto& kv1 : m_unigrams)
	{
		Unigram& unigram = kv1.second;
		for (auto& kv2 : unigram.bigrams)
		{
			Bigram& bigram = kv2.second;
			bigram.prob = (bigram.count + m_addK) / (unigram.count + m_addK*m_unigrams.size());
		}
	}
}


std::vector<uint32_t> LanguageModel::utf8ToCodepoint(const std::string& s)
{
	std::vector<uint32_t> res;
	auto iter = s.begin();
	auto end = s.end();
	while (iter != end)
	{
		uint32_t c = utf8::next(iter, end);
		res.push_back(c);
	}

	return res;
}


std::unordered_map<uint32_t, uint32_t> LanguageModel::codepointToLabelMapping(const std::vector<uint32_t>& charsCP)
{
	std::unordered_map<uint32_t, uint32_t> res;
	for (size_t i = 0; i < charsCP.size(); ++i)
	{
		res[charsCP[i]] = static_cast<uint32_t>(i);
	}

	return res;
}


std::vector<uint32_t> LanguageModel::utf8ToLabel(const std::string& utf8Str)
{
	std::vector<uint32_t> res;
	auto iter = utf8Str.begin();
	auto end = utf8Str.end();
	while (iter != end)
	{
		uint32_t c = utf8::next(iter, end);
		res.push_back(m_codepointToLabel[c]);
	}

	return res;
}


std::string LanguageModel::labelToUtf8(const std::vector<uint32_t>& labelStr)
{
	std::string res;
	for (const auto c : labelStr)
	{
		utf8::append(m_labelToCodepoint[c], std::back_inserter(res));
	}

	return res;
}


double LanguageModel::getUnigramProb(const std::vector<uint32_t>& w) const
{
	// get entry for w
	const auto iter = m_unigrams.find(w);
	if (iter == m_unigrams.end())
	{
		return 0.0;
	}

	// return unigram prob
	return iter->second.prob;
}


double LanguageModel::getBigramProb(const std::vector<uint32_t>& w1, const std::vector<uint32_t>& w2) const
{
	// get entry for w1
	const auto iter1 = m_unigrams.find(w1);
	if (iter1 == m_unigrams.end())
	{
		return 0.0;
	}

	// get entry for w2
	const Unigram& unigram = iter1->second;
	const auto iter2 = unigram.bigrams.find(w2);
	if (iter2 == unigram.bigrams.end())
	{
		return m_addK / (unigram.count + m_addK*m_unigrams.size());
	}

	// return bigram prob
	return iter2->second.prob;
}


bool LanguageModel::isWord(const std::vector<uint32_t>& text) const 
{
	return m_tree.isWord(text); 
}


std::vector<std::vector<uint32_t>> LanguageModel::getNextWords(const std::vector<uint32_t>& text) const 
{
	return m_tree.getNextWords(text); 
}


std::vector<uint32_t> LanguageModel::getNextChars(const std::vector<uint32_t>& text) const
{
	// query tree
	std::vector<uint32_t> res(m_tree.getNextChars(text));
	
	// if between words or if word is complete, then add non word chars
	if (text.empty() || isWord(text))
	{
		res.insert(res.end(), m_nonWordLabels.begin(), m_nonWordLabels.end());
	}

	return res;
}


void LanguageModel::initLabelSets(const std::unordered_map<uint32_t, uint32_t>& codepointToLabelMapping, const std::vector<uint32_t>& wordCodepoints)
{
	for (const auto kv : codepointToLabelMapping)
	{
		const uint32_t codepoint = kv.first;
		const uint32_t label = kv.second;

		// word char
		if (std::find(wordCodepoints.begin(), wordCodepoints.end(), codepoint)!= wordCodepoints.end())
		{
			m_wordLabels.insert(label);
		}
		// non word char
		else
		{
			m_nonWordLabels.insert(label);
		}

		m_allLabels.insert(label);
	}
}


const std::set<uint32_t>& LanguageModel::getAllChars() const
{
	return m_allLabels;
}


const std::set<uint32_t>& LanguageModel::getWordChars() const
{
	return m_wordLabels;;
}


const std::set<uint32_t>& LanguageModel::getNonWordChars() const
{
	return m_nonWordLabels;
}


