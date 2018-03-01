#pragma once
#include <cstddef>


// matrix interface
class IMatrix
{
public:
	virtual double getAt(size_t row, size_t col) const = 0;
	virtual void setAt(size_t row, size_t col, double val) = 0;
	size_t rows() const { return m_rows; }
	size_t cols() const { return m_cols; }

protected:
	size_t m_rows;
	size_t m_cols;
};

