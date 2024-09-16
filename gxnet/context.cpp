
#include "context.h"

namespace gxnet {

ForwardCtx :: ForwardCtx( const DataVector * input )
{
	mInput = input;
}

ForwardCtx :: ~ForwardCtx()
{
}

void ForwardCtx :: setInput( const DataVector * input )
{
	mInput = input;
}

const DataVector & ForwardCtx :: getInput()
{
	return *mInput;
}

DataVector & ForwardCtx :: getOutput()
{
	return mOutput;
}

////////////////////////////////////////////////////////////

BackwardCtx :: BackwardCtx()
{
}

BackwardCtx :: BackwardCtx( const BackwardCtx & other )
{
	mDelta.resize( other.getDelta().size() );

	for( auto & item : other.getGradients() ) {
		mGradients.emplace_back( DataVector( item.size() ) );
	}
}

BackwardCtx :: ~BackwardCtx()
{
}

DataVector & BackwardCtx :: getDelta()
{
	return mDelta;
}

const DataVector & BackwardCtx :: getDelta() const
{
	return mDelta;
}

DataMatrix & BackwardCtx :: getGradients()
{
	return mGradients;
}

const DataMatrix & BackwardCtx :: getGradients() const
{
	return mGradients;
}


////////////////////////////////////////////////////////////

BaseLayerCtx :: BaseLayerCtx( const DataVector * input )
	: mForwardCtx( input )
{
}

BaseLayerCtx :: ~BaseLayerCtx()
{
}

ForwardCtx & BaseLayerCtx :: getForwardCtx()
{
	return mForwardCtx;
}

BackwardCtx & BaseLayerCtx :: getBackwardCtx()
{
	return mBackwardCtx;
}

}; // namespace gxnet;

