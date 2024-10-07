#pragma once

#include "common.h"

#include <vector>

namespace gxnet {

class BaseLayerContext;
typedef std::vector< BaseLayerContext * > BaseLayerContextPtrVector;

class BackwardContext;
typedef std::vector< BackwardContext * > BackwardContextPtrVector;

class BackwardContext {
public:
	BackwardContext();

	virtual ~BackwardContext();

	MDSpanRW & getDeltaMS();

	const MDSpanRO & getDeltaRO() const;

	DataMatrix & getGradients();

	const DataMatrix & getGradients() const;

protected:
	DataMatrix mGradients;

	DataVector mDelta;
	MDSpanRW mDeltaMS;

	MDSpanRO mDeltaRO;
};

class BaseLayerContext : public BackwardContext {
public:
	BaseLayerContext();

	virtual ~BaseLayerContext();

	void setInMS( const MDSpanRO * inMS );

	const MDSpanRO & getInMS();

	MDSpanRW & getOutMS();

	const MDSpanRO & getOutRO();

protected:
	const MDSpanRO * mInMS;

	MDSpanRW mOutMS;

	MDSpanRO mOutRO;

	DataVector mOutput;
};

class FullConnLayerContext : public BaseLayerContext {
public:
	FullConnLayerContext();

	virtual ~FullConnLayerContext();

	DataVector & getTempWeights();

	DataVector & getTempGradients();

protected:
	DataVector mTempWeights, mTempGradients;
};

class ConvExLayerContext : public BaseLayerContext {
public:
	ConvExLayerContext();
	~ConvExLayerContext();

	DataMatrix & getRows4collectGradients();

	DataMatrix & getRows4calcOutput();

	DataMatrix & getRows4backpropagate();

	DataMatrix & getRowsOfDelta();

	DataVector & getPaddingDelta();

private:
	DataMatrix mRows4collectGradient, mRows4calcOutput,
			mRows4backpropagate, mRowsOfDelta;
	DataVector mPaddingDelta;
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

