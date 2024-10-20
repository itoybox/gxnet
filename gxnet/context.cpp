
#include "context.h"

namespace gxnet {

BackwardContext :: BackwardContext()
{
}

BackwardContext :: ~BackwardContext()
{
}

MDVector & BackwardContext :: getDeltaMD()
{
	return mDeltaMD;
}

const MDVector & BackwardContext :: getDeltaMD() const
{
	return mDeltaMD;
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
{
	mInMD = NULL;
}

BaseLayerContext :: ~BaseLayerContext()
{
}

void BaseLayerContext :: setInMD( const MDVector * inMD )
{
	mInMD = inMD;
}

const MDVector & BaseLayerContext :: getInMD()
{
	return * mInMD;
}

MDVector & BaseLayerContext :: getOutMD()
{
	return mOutMD;
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

ConvLayerContext :: ConvLayerContext()
{
}

ConvLayerContext :: ~ConvLayerContext()
{
}

MDVector & ConvLayerContext :: getPaddingDeltaMD()
{
	return mPaddingDeltaMD;
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

