#pragma once
#include "HashFunction.hpp"
#include "LanguageModel.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <limits>
#include <stdint.h>
#include <cstddef>


class Beam
{
public:
	// CTOR
	Beam(const std::shared_ptr<LanguageModel>& lm, bool useNGrams, bool forcastNGrams, bool sampleNGrams);

	// next possible characters and words
	const std::vector<uint32_t>& getText() const;
	std::vector<uint32_t> getNextChars() const;

	// create child beam by extending by given character
	std::shared_ptr<Beam> createChildBeam(double prBlank, double prNonBlank, uint32_t newChar=std::numeric_limits<uint32_t>::max()) const;

	// merge given beam with this beam
	void mergeBeam(const std::shared_ptr<Beam>& beam);

	// complete the text (last word) of the beam
	void completeText();

	// get probabilities of beam
	double getBlankProb() const { return m_prBlank; } // optical: paths ending with blank
	double getNonBlankProb() const { return m_prNonBlank; } // optical: paths ending with non-blank
	double getTotalProb() const { return m_prBlank + m_prNonBlank; } // optical: total
	double getTextualProb() const { return m_prTextTotal; } // textual

private:
	std::shared_ptr<LanguageModel> m_lm;

	// optical part
	double m_prBlank = 1.0;
	double m_prNonBlank = 0.0;

	// textual part
	std::vector<uint32_t> m_text; // complete text of this beam
	std::vector<uint32_t> m_wordDev; // currently "built" word
	std::vector<std::vector<uint32_t>> m_wordHist; // history of words in text
	double m_prTextTotal = 1.0;
	double m_prTextUnnormalized = 1.0;
	bool m_useNGrams = false;
	bool m_forcastNGrams = false;
	bool m_sampleNGrams = false;

	// methods to score beam text by LM
	void handleNGrams(std::shared_ptr<Beam>& newBeam, uint32_t newChar) const;
	std::pair<double, std::vector<std::vector<uint32_t>>> getNextWordsSampled(const std::shared_ptr<LanguageModel>& lm, const std::vector<uint32_t>& text) const;
};


// holds all beams at one time-step
class BeamList
{
public:
	// add beam to list
	void addBeam(const std::shared_ptr<Beam>& beam);

	// sort beams according to (totalProb*textualProb) and return best beams
	std::vector<std::shared_ptr<Beam>> getBestBeams(size_t beamWidth);

private:
	std::unordered_map<std::vector<uint32_t>, std::shared_ptr<Beam>, HashFunction> m_beams;
};

