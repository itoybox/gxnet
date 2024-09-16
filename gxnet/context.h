#pragma once

#include "common.h"

#include <vector>

namespace gxnet {

class BaseLayerCtx;
typedef std::vector< BaseLayerCtx * > BaseLayerCtxPtrVector;

class BackwardCtx;
typedef std::vector< BackwardCtx * > BackwardCtxPtrVector;

class ForwardCtx {
public:
	ForwardCtx( const DataVector * input );

	virtual ~ForwardCtx();

	void setInput( const DataVector * input );

	const DataVector & getInput();

	DataVector & getOutput();

private:
	const DataVector * mInput;
	DataVector mOutput;
};

class BackwardCtx {
public:
	BackwardCtx();

	BackwardCtx( const BackwardCtx & other );

	~BackwardCtx();

	DataVector & getDelta();

	const DataVector & getDelta() const;

	DataMatrix & getGradients();

	const DataMatrix & getGradients() const;

private:
	DataVector mDelta;
	DataMatrix mGradients;
};

class BaseLayerCtx {
public:
	BaseLayerCtx( const DataVector * input );

	virtual ~BaseLayerCtx();

	ForwardCtx & getForwardCtx();

	BackwardCtx & getBackwardCtx();

private:
	ForwardCtx mForwardCtx;
	BackwardCtx mBackwardCtx;
};

}; // namespace gxnet;

