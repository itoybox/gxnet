
#include "context.h"

namespace gxnet {

BackwardContext :: BackwardContext()
	: mDeltaMS( mDelta ), mDeltaRO( mDeltaMS )
{
}

BackwardContext :: ~BackwardContext()
{
}

MDSpanRW & BackwardContext :: getDeltaMS()
{
	return mDeltaMS;
}

const MDSpanRO & BackwardContext :: getDeltaRO() const
{
	return mDeltaRO;
}

DataMatrix & BackwardContext :: getGradients()
{
	return mGradients;
}

const DataMatrix & BackwardContext :: getGradients() const
{
	return mGradients;
}

////////////////////////////////////////////////////////////

BaseLayerContext :: BaseLayerContext()
	: mOutMS( mOutput ), mOutRO( mOutMS )
{
	mInMS = NULL;
}

BaseLayerContext :: ~BaseLayerContext()
{
}

void BaseLayerContext :: setInMS( const MDSpanRO * inMS )
{
	mInMS = inMS;
}

const MDSpanRO & BaseLayerContext :: getInMS()
{
	return *mInMS;
}

MDSpanRW & BaseLayerContext :: getOutMS()
{
	return mOutMS;
}

const MDSpanRO & BaseLayerContext :: getOutRO()
{
	return mOutRO;
}

////////////////////////////////////////////////////////////

FullConnLayerContext :: FullConnLayerContext()
{
}

FullConnLayerContext :: ~FullConnLayerContext()
{
}

DataVector & FullConnLayerContext :: getTempWeights()
{
	return mTempWeights;
}

DataVector & FullConnLayerContext :: getTempGradients()
{
	return mTempGradients;
}

////////////////////////////////////////////////////////////

ConvExLayerContext :: ConvExLayerContext()
{
}

ConvExLayerContext :: ~ConvExLayerContext()
{
}

DataMatrix & ConvExLayerContext :: getRows4collectGradients()
{
	return mRows4collectGradient;
}

DataMatrix & ConvExLayerContext :: getRows4calcOutput()
{
	return mRows4calcOutput;
}

DataMatrix & ConvExLayerContext :: getRows4backpropagate()
{
	return mRows4backpropagate;
}

DataMatrix & ConvExLayerContext :: getRowsOfDelta()
{
	return mRowsOfDelta;
}

DataVector & ConvExLayerContext :: getPaddingDelta()
{
	return mPaddingDelta;
}

////////////////////////////////////////////////////////////

DropoutLayerContext :: DropoutLayerContext()
{
}

DropoutLayerContext :: ~DropoutLayerContext()
{
}

BoolVector & DropoutLayerContext :: getMask()
{
	return mMask;
}


}; // namespace gxnet;

