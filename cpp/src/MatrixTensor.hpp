#pragma once
#include "IMatrix.hpp"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>


template<class T>
class MatrixTensor : public IMatrix
{
public:
	MatrixTensor(const T& tensor, size_t b, size_t maxT, size_t maxC)
	:m_tensor(tensor)
	,m_batch(b)
	{
		m_rows=maxT;
		m_cols=maxC;
	}

	virtual double getAt(size_t row, size_t col) const
	{
		using namespace tensorflow;
		return m_tensor((int32)row, (int32)m_batch, (int32)col);
	}
	
	virtual void setAt(size_t row, size_t col, double val)
	{
		// not implemented
	}

private:
	const T& m_tensor;
	size_t m_batch;
};

