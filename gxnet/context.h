#pragma once

#include "common.h"

#include <vector>
#include <utility>

namespace gxnet {

class BaseLayerContext;
typedef std::vector< BaseLayerContext * > BaseLayerContextPtrVector;

class BackwardContext;
typedef std::vector< BackwardContext * > BackwardContextPtrVector;

class BackwardContext {
public:
	BackwardContext();

	virtual ~BackwardContext();

	MDVector & getDelta();

	const MDVector & getDelta() const;

	MDVector & getGradients();

	const MDVector & getGradients() const;

protected:
	MDVector mGradients;

	MDVector mDelta;
};

class BaseLayerContext : public BackwardContext {
public:
	BaseLayerContext();

	virtual ~BaseLayerContext();

	void setInput( const MDVector * input );

	const MDVector & getInput();

	MDVector & getOutput();

protected:
	const MDVector * mInput;

	MDVector mOutput;
};

class FullConnLayerContext : public BaseLayerContext {
public:
	FullConnLayerContext();

	virtual ~FullConnLayerContext();

	DataVector & getTempGradients();

protected:
	DataVector mTempGradients;
};

class ConvLayerContext : public BaseLayerContext {
public:
	ConvLayerContext();
	~ConvLayerContext();

	MDVector & getPaddingDelta();

private:
	MDVector mPaddingDelta;
};

class ConvExLayerContext : public ConvLayerContext {
public:
	ConvExLayerContext();
	~ConvExLayerContext();

	MDVector & getRows4collectGradients();

	MDVector & getRows4calcOutput();

	MDVector & getRows4backpropagate();

	DataVector & getTempGradients();

private:
	MDVector mRows4calcOutput, mRows4backpropagate, mRows4collectGradient;
	DataVector mTempGradients;
};

class DropoutLayerContext : public BaseLayerContext {
public:
	DropoutLayerContext();
	~DropoutLayerContext();

	BoolVector & getMask();

private:
	mutable BoolVector mMask;
};

}; // namespace gxnet;

