#pragma once

#include "common.h"

namespace gxnet {

class Optim {
public:
	enum { eSGD = 1 };

public:
	Optim( int type );

	virtual ~Optim();

	void setDebug( bool isDebug );

	int getType() const;

	virtual void update( DataVector * weights, const DataVector & gradients,
			size_t trainingCount, size_t miniBatchCount ) = 0;

	virtual void updateBiases( DataVector * bias, const DataVector & delta,
			size_t miniBatchCount ) = 0;

public:

	static Optim * SGD( DataType lr, DataType lambda );

protected:
	bool mIsDebug;
	int mType;
};


}; // namespace gxnet;

