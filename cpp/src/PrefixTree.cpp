#include "PrefixTree.hpp"
#include <deque>
#include <algorithm>


PrefixTree::PrefixTree()
:m_root(std::make_shared<Node>())
{
}


void PrefixTree::addWord(const std::vector<uint32_t>& word)
{
	std::shared_ptr<Node> node = m_root;
	const size_t len = word.size();
	for (size_t i = 0; i < len; ++i)
	{
		const uint32_t c = word[i];
		auto iter = std::find_if(node->children.begin(), node->children.end(), [&](const std::pair<uint32_t, std::shared_ptr<Node>>& p) {return p.first == c; });
		if (iter == node->children.end())
		{
			std::shared_ptr<Node> newNode = std::make_shared<Node>();
			node->children.push_back(std::make_pair(c, newNode));
			node = newNode;
		}
		else
		{
			node = iter->second;
		}

		if (i + 1 == len)
		{
			node->word = word;
		}
	}
}


void PrefixTree::addWords(const std::vector<std::vector<uint32_t>>& words)
{
	for (const auto& w : words)
	{
		addWord(w);
	}
}


void PrefixTree::allWordsAdded()
{
	std::deque<std::shared_ptr<Node>> nodes = { m_root };
	while (!nodes.empty())
	{
		// current node
		auto& node = nodes.front();

		// sort children by label to allow binary search
		std::sort
		(
			node->children.begin()
			,node->children.end()
			,[](const std::pair<uint32_t, std::shared_ptr<Node>>& lhs, const std::pair<uint32_t, std::shared_ptr<Node>>& rhs) {return lhs.first < rhs.first; }
		);

		// go over all child nodes
		for (const auto& kv : node->children)
		{
			// add node
			nodes.push_back(kv.second);
		}

		// remove current node from queue
		nodes.pop_front();
	}
}


bool PrefixTree::isWord(const std::vector<uint32_t>& text) const
{
	std::shared_ptr<Node> node = getNode(text);
	if (!node)
	{
		return false;
	}

	return !node->word.empty();
}


std::vector<uint32_t> PrefixTree::getNextChars(const std::vector<uint32_t>& text) const
{
	std::vector<uint32_t> res;
	std::shared_ptr<Node> node = getNode(text);
	if (!node)
	{
		return res;
	}

	for (const auto kv : node->children)
	{
		res.push_back(kv.first);
	}

	return res;
}


std::vector<std::vector<uint32_t>> PrefixTree::getNextWords(const std::vector<uint32_t>& text) const
{
	// search start node, that is the node representing the given prefix
	std::vector<std::vector<uint32_t>> res;
	const auto startNode = getNode(text);
	if (!startNode)
	{
		return res;
	}

	// cache for level 1
	const bool isLevel1 = text.size() == 1;
	const auto cacheIter = m_level1Cache.find(text[0]);
	if (isLevel1 && cacheIter != m_level1Cache.end())
	{
		return cacheIter->second;
	}
	
	// search all words starting with the given prefix
	std::deque<std::shared_ptr<Node>> nodes = { startNode };
	while (!nodes.empty())
	{
		// current node
		const auto& node=nodes.front();

		// go over all child nodes
		for (const auto& kv : node->children)
		{
			// add node
			nodes.push_back(kv.second);
		}

		// remember current prefix if it is a word
		if (!node->word.empty())
		{
			res.push_back(node->word);
		}
		
		// remove current node from queue
		nodes.pop_front();
	}

	// put result for level 1 into cache
	if (isLevel1)
	{
		m_level1Cache[text[0]] = res;
	}

	return res;
}


std::shared_ptr<PrefixTree::Node> PrefixTree::getNode(const std::vector<uint32_t>& text) const
{
	// start with root
	std::shared_ptr<Node> node = m_root;
	for (const auto c : text)
	{
		// find child element representing current char (binary search)
		auto iter = std::lower_bound(node->children.begin(), node->children.end(), c, [](const std::pair<uint32_t, std::shared_ptr<Node>>& p, uint32_t val) {return p.first < val; });
		if (iter == node->children.end() || iter->first > c)
		{
			// not found
			return std::shared_ptr<PrefixTree::Node>();
		}

		// continue with the child node
		node = iter->second;
	}

	return node;
}


