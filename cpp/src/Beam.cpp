#include "Beam.hpp"
#include <cassert>
#include <algorithm>
#include <math.h>
#include <iostream>


Beam::Beam(const std::shared_ptr<LanguageModel>& lm, bool useNGrams, bool forcastNGrams, bool sampleNGrams)
:m_lm(lm)
,m_useNGrams(useNGrams)
,m_forcastNGrams(forcastNGrams)
,m_sampleNGrams(sampleNGrams)
{
}


const std::vector<uint32_t>& Beam::getText() const
{
	return m_text;
}


std::vector<uint32_t> Beam::getNextChars() const
{
	return m_lm->getNextChars(m_wordDev);
}


std::pair<double, std::vector<std::vector<uint32_t>>> Beam::getNextWordsSampled(const std::shared_ptr<LanguageModel>& lm, const std::vector<uint32_t>& text) const
{
	const size_t maxSampleSize = 20;
	auto nextWords=lm->getNextWords(text);

	// if sampling not enabled or sampling not needed (too few words), then return all words
	if (!m_sampleNGrams || nextWords.size()<maxSampleSize)
	{
		return std::make_pair(1.0, nextWords);
	}

	// take random sample, adjust factor which is used to correct N-gram probability
	const double factor = double(nextWords.size())/double(maxSampleSize);
	std::random_shuffle(nextWords.begin(), nextWords.end());
	nextWords.resize(maxSampleSize);
	return std::make_pair(factor, nextWords);
}


void Beam::handleNGrams(std::shared_ptr<Beam>& newBeam, uint32_t newChar) const
{
	const auto& wordChars(newBeam->m_lm->getWordChars());

	// char occurs inside a word
	if (wordChars.find(newChar) != wordChars.end())
	{
		newBeam->m_wordDev.push_back(newChar);
		
		// get next words, possibly sampled
		if(m_forcastNGrams)
		{
			std::vector<std::vector<uint32_t>> nextWords;
			double sampleFactor = 1.0;
			std::tie(sampleFactor, nextWords) = getNextWordsSampled(newBeam->m_lm, newBeam->m_wordDev);
		
			// sum over all unigram/bigram probabilities
			const size_t numWords = newBeam->m_wordHist.size();
			double sum = 0.0;
			if (numWords == 0)
			{
				for (const auto& w : nextWords)
				{
					sum += newBeam->m_lm->getUnigramProb(w);
				}
			}
			else
			{
				const auto& lastWord = newBeam->m_wordHist.back();
				for (const auto& w : nextWords)
				{
					sum += newBeam->m_lm->getBigramProb(lastWord, w);
				}
			}

			// correct sampling 
			sum = std::min(sum*sampleFactor, 1.0);

			// set calculated probability
			newBeam->m_prTextTotal = newBeam->m_prTextUnnormalized*sum;
			newBeam->m_prTextTotal = numWords >= 1 ? pow(newBeam->m_prTextTotal, 1.0 / (numWords + 1)) : newBeam->m_prTextTotal;
		}
	}
	else
	{
		// current word not empty
		if (!newBeam->m_wordDev.empty())
		{
			newBeam->m_wordHist.push_back(newBeam->m_wordDev);
			newBeam->m_wordDev.clear();

			const size_t numWords = newBeam->m_wordHist.size();
			if (numWords == 1)
			{
				newBeam->m_prTextUnnormalized *= newBeam->m_lm->getUnigramProb(newBeam->m_wordHist.back());
				newBeam->m_prTextTotal = newBeam->m_prTextUnnormalized;
			}
			else if (numWords >= 2)
			{
				newBeam->m_prTextUnnormalized *= newBeam->m_lm->getBigramProb(*(newBeam->m_wordHist.end() - 2), *(newBeam->m_wordHist.end() - 1));
				newBeam->m_prTextTotal = pow(newBeam->m_prTextUnnormalized, 1.0 / numWords);
			}
		}

	}
}


std::shared_ptr<Beam> Beam::createChildBeam(double prBlank, double prNonBlank, uint32_t newChar) const
{
	// copy this beam
	std::shared_ptr<Beam> newBeam = std::make_shared<Beam>(*this);

	// add new char to text and assign calculated probabilities
	if (newChar != std::numeric_limits<uint32_t>::max())
	{
		if (m_useNGrams)
		{
			handleNGrams(newBeam, newChar);
		}
		else
		{
			const auto& wordChars(newBeam->m_lm->getWordChars());
			if (wordChars.find(newChar) != wordChars.end())
			{
				newBeam->m_wordDev.push_back(newChar);
			}
			else
			{
				newBeam->m_wordDev.clear();
			}
		}
		
		// always append new char to text of beam
		newBeam->m_text.push_back(newChar);
	}
	
	newBeam->m_prBlank = prBlank;
	newBeam->m_prNonBlank = prNonBlank;
	return newBeam;
}


void Beam::mergeBeam(const std::shared_ptr<Beam>& beam)
{
	assert(getText() == beam->getText());
	
	// sum up probabilities
	m_prBlank += beam->m_prBlank;
	m_prNonBlank += beam->m_prNonBlank;
}


void Beam::completeText()
{
	// nothing to do if beam has no unfinished words at the end
	if (m_wordDev.empty())
	{
		return;
	}

	// get next words
	const auto nextWords=m_lm->getNextWords(m_wordDev);

	// if only one next word possible, then take this word and complete beam with it
	if (nextWords.size() == 1)
	{
		assert(m_text.size()>=m_wordDev.size());
		const auto& completeWord = nextWords[0];
		m_text.resize(m_text.size() - m_wordDev.size());
		m_text.insert(m_text.end(), completeWord.begin(), completeWord.end());
	}
	
}


void BeamList::addBeam(const std::shared_ptr<Beam>& beam)
{
	// if beam text already in list, merge beams, otherwise add new beam
	auto iter=m_beams.find(beam->getText());
	if (iter == m_beams.end())
	{
		m_beams[beam->getText()] = beam;
	}
	else
	{
		iter->second->mergeBeam(beam);
	}
}


std::vector<std::shared_ptr<Beam>> BeamList::getBestBeams(size_t beamWidth)
{
	// sort by totalProb*textualProb
	typedef std::pair<std::vector<uint32_t>, std::shared_ptr<Beam>> KeyValueType;
	std::vector<KeyValueType> beams(m_beams.begin(), m_beams.end());
	std::sort
	(
		beams.begin()
		,beams.end()
		,[](const KeyValueType& a, const KeyValueType& b) {return a.second->getTotalProb()*a.second->getTextualProb() > b.second->getTotalProb()*b.second->getTextualProb(); }
	);

	// take beam object (take value, ignore key) and return it
	std::vector<std::shared_ptr<Beam>> res;
	res.reserve(beamWidth);
	for (size_t i = 0; i < beams.size() && i < beamWidth; ++i)
	{
		res.push_back(beams[i].second);
	}
	return res;
}

