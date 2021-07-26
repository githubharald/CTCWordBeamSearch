#include "Metrics.hpp"
#include <algorithm>
#include <iostream>
#include <map>


Metrics::Metrics(const std::set<uint32_t>& wordChars)
:m_wordChars(wordChars)
{
}


std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Metrics::getWordIDStrings(const std::vector<uint32_t>& t1, const std::vector<uint32_t>& t2) const
{
	std::pair<std::vector<uint32_t>, std::vector<uint32_t>> res;
	std::map<std::vector<uint32_t>, uint32_t> wordIDs;

	std::vector<uint32_t> currWord;
	uint32_t currID = 0;

	// first text
	for (size_t i = 0; i < t1.size(); ++i)
	{
		const uint32_t c = t1[i];
		
		// if its a word-char
		if (m_wordChars.find(c) != m_wordChars.end())
		{
			currWord.push_back(c);
		}

		// if it is a non-word-char, or if it the last char in the text
		if(!(m_wordChars.find(c) != m_wordChars.end()) || i+1==t1.size())
		{
			// is word not empty
			if (!currWord.empty())
			{
				// word not yet known, assign an ID
				if (wordIDs.find(currWord) == wordIDs.end())
				{
					wordIDs[currWord] = currID;
					++currID;
				}

				res.first.push_back(wordIDs[currWord]);
				currWord.clear();
			}
		}
	}

	// reset current word for next text
	currWord.clear();

	// second text
	for (size_t i = 0; i < t2.size(); ++i)
	{
		const uint32_t c = t2[i];

		// if its a word-char
		if (m_wordChars.find(c) != m_wordChars.end())
		{
			currWord.push_back(c);
		}

		// if it is a non-word-char, or if it the last char in the text
		if (!(m_wordChars.find(c) != m_wordChars.end()) || i + 1 == t2.size())
		{
			// is word not empty
			if (!currWord.empty())
			{
				// word not yet known, assign an ID
				if (wordIDs.find(currWord) == wordIDs.end())
				{
					wordIDs[currWord] = currID;
					++currID;
				}

				res.second.push_back(wordIDs[currWord]);
				currWord.clear();
			}
		}
	}

	return res;
}


void Metrics::addResult(const std::vector<uint32_t>& gt, const std::vector<uint32_t>& rec)
{
	// CER
	m_numChars += gt.size();
	m_edChars += editDistance(gt, rec);

	// WER
	const auto idStrings = getWordIDStrings(gt, rec);
	m_numWords += idStrings.first.size();
	m_edWords += editDistance(idStrings.first, idStrings.second);
}


size_t Metrics::editDistance(const std::vector<uint32_t>& t1, const std::vector<uint32_t>& t2)
{
	// taken from: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
	const std::size_t len1 = t1.size(), len2 = t2.size();
	std::vector<size_t> col(len2 + 1), prevCol(len2 + 1);

	for (size_t i = 0; i < prevCol.size(); i++)
	{
		prevCol[i] = i;
	}
		
	for (size_t i = 0; i < len1; i++)
	{
		col[0] = i + 1;
		for (size_t j = 0; j < len2; j++)
		{
			col[j + 1] = std::min({ prevCol[1 + j] + 1, col[j] + 1, prevCol[j] + (t1[i] == t2[j] ? 0 : 1) });
		}
		col.swap(prevCol);
	}
	return prevCol[len2];
}


double Metrics::getCER() const
{
	return m_numChars > 0 ? double(m_edChars) / double(m_numChars) : 0.0;
}


double Metrics::getWER() const
{
	return m_numWords > 0 ? double(m_edWords) / double(m_numWords) : 0.0;
}

