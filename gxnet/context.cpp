
#include "context.h"

namespace gxnet {

BackwardContext :: BackwardContext()
{
}

BackwardContext :: ~BackwardContext()
{
}

MDVector & BackwardContext :: getDelta()
{
	return mDelta;
}

const MDVector & BackwardContext :: getDelta() const
{
	return mDelta;
}

MDVector & BackwardContext :: getGradients()
{
	return mGradients;
}

const MDVector & BackwardContext :: getGradients() const
{
	return mGradients;
}

////////////////////////////////////////////////////////////

BaseLayerContext :: BaseLayerContext()
{
	mInput = NULL;
}

BaseLayerContext :: ~BaseLayerContext()
{
}

void BaseLayerContext :: setInput( const MDVector * input )
{
	mInput = input;
}

const MDVector & BaseLayerContext :: getInput()
{
	return * mInput;
}

MDVector & BaseLayerContext :: getOutput()
{
	return mOutput;
}

////////////////////////////////////////////////////////////

FullConnLayerContext :: FullConnLayerContext()
{
}

FullConnLayerContext :: ~FullConnLayerContext()
{
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

MDVector & ConvLayerContext :: getPaddingDelta()
{
	return mPaddingDelta;
}

////////////////////////////////////////////////////////////

ConvExLayerContext :: ConvExLayerContext()
{
}

ConvExLayerContext :: ~ConvExLayerContext()
{
}

MDVector & ConvExLayerContext :: getRows4collectGradients()
{
	return mRows4collectGradient;
}

MDVector & ConvExLayerContext :: getRows4calcOutput()
{
	return mRows4calcOutput;
}

MDVector & ConvExLayerContext :: getRows4backpropagate()
{
	return mRows4backpropagate;
}

DataVector & ConvExLayerContext :: getTempGradients()
{
	return mTempGradients;
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

