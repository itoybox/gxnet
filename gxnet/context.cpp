
#include "context.h"

namespace gxnet {

BackwardContext :: BackwardContext()
{
}

BackwardContext :: ~BackwardContext()
{
}

DataVector & BackwardContext :: getDelta()
{
	return mDelta;
}

const DataVector & BackwardContext :: getDelta() const
{
	return mDelta;
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

}; // namespace gxnet;

