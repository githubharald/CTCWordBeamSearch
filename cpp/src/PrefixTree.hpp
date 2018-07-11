#pragma once
#include <vector>
#include <memory>
#include <map>
#include <stdint.h>
#include <cstddef>


// prefix tree which allows querying next possible characters and words for a given text
class PrefixTree
{
public:
	// CTOR
	PrefixTree();

	// add words to the prefix tree. After all words are added, allWordsAdded() must be called to setup search structures.
	void addWord(const std::vector<uint32_t>& word);
	void addWords(const std::vector<std::vector<uint32_t>>& words);
	void allWordsAdded();

	// query prefix tree
	bool isWord(const std::vector<uint32_t>& text) const;
	std::vector<uint32_t> getNextChars(const std::vector<uint32_t>& text) const;
	std::vector<std::vector<uint32_t>> getNextWords(const std::vector<uint32_t>& text) const;


private:
	// node of the prefix tree
	struct Node
	{
		std::vector<std::pair<uint32_t, std::shared_ptr<Node>>> children;
		std::vector<uint32_t> word;
	};

	std::shared_ptr<Node> m_root; // the root represents the empty text
	std::shared_ptr<Node> getNode(const std::vector<uint32_t>& text) const; // get the node for a given text
	mutable std::map<uint32_t, std::vector<std::vector<uint32_t>>> m_level1Cache; // cache words of nodes in level 1
};
