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

	DataVector & getDelta();

	const DataVector & getDelta() const;

	DataMatrix & getGradients();

	const DataMatrix & getGradients() const;

protected:
	DataVector mDelta;
	DataMatrix mGradients;
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

}; // namespace gxnet;

