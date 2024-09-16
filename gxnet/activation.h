#pragma once

#include "common.h"

namespace gxnet {

class ActFunc {
public:
	enum { eSigmoid = 1, eLeakyReLU = 2, eTanh = 3, eSoftmax = 4 };
public:
	ActFunc( int type );

	~ActFunc();

	int getType() const;

	void activate( const DataVector & input, DataVector * output ) const;

	void derivate( const DataVector & output, DataVector * outDelta ) const;

public:

	static ActFunc * sigmoid();

	static ActFunc * tanh();

	static ActFunc * leakyReLU();

	static ActFunc * softmax();

private:
	int mType;
};


}; // namespace gxnet;

