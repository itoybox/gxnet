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

	MDVector & getDeltaMD();

	const MDVector & getDeltaMD() const;

	DataMatrix & getGradients();

	const DataMatrix & getGradients() const;

protected:
	DataMatrix mGradients;

	MDVector mDeltaMD;
};

class BaseLayerContext : public BackwardContext {
public:
	BaseLayerContext();

	virtual ~BaseLayerContext();

	void setInMD( const MDVector * inMD );

	const MDVector & getInMD();

	MDVector & getOutMD();

protected:
	const MDVector * mInMD;

	MDVector mOutMD;
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

class ConvLayerContext : public BaseLayerContext {
public:
	ConvLayerContext();
	~ConvLayerContext();

	MDVector & getPaddingDeltaMD();

private:
	MDVector mPaddingDeltaMD;
};

class ConvExLayerContext : public ConvLayerContext {
public:
	ConvExLayerContext();
	~ConvExLayerContext();

	DataMatrix & getRows4collectGradients();

	DataMatrix & getRows4calcOutput();

	DataMatrix & getRows4backpropagate();

	DataMatrix & getRowsOfDelta();

private:
	DataMatrix mRows4collectGradient, mRows4calcOutput,
			mRows4backpropagate, mRowsOfDelta;
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

