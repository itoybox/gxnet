#include "layer.h"
#include "context.h"

#include "utils.h"
#include "activation.h"
#include "optim.h"

#include <limits.h>
#include <cstdio>

#include <iostream>

namespace gxnet {

BaseLayer :: BaseLayer( int type )
{
	mType = type;
	mActFunc = NULL;

	mIsTraining = false;
}

BaseLayer :: ~BaseLayer()
{
	if( NULL != mActFunc ) delete mActFunc;
}

void BaseLayer :: forward( BaseLayerCtx * ctx ) const
{
	assert( ctx->getForwardCtx().getInput().size() == getInputSize() );

	ForwardCtx & fwdCtx = ctx->getForwardCtx();

	calcOutput( ctx );
	if( NULL != mActFunc ) {
		if( gx_is_inner_debug ) Utils::printVector( "before.act", fwdCtx.getOutput() );

		mActFunc->activate( fwdCtx.getOutput(), &( fwdCtx.getOutput() ) );

		if( gx_is_inner_debug ) Utils::printVector( "after.act", fwdCtx.getOutput() );
	}
}

void BaseLayer :: backward( BaseLayerCtx * ctx, DataVector * inDelta ) const
{
	ForwardCtx & fwdCtx = ctx->getForwardCtx();
	BackwardCtx & bwdCtx = ctx->getBackwardCtx();

	if( NULL != mActFunc ) mActFunc->derivate( fwdCtx.getOutput(), &( bwdCtx.getDelta() ) );

	if( NULL != inDelta ) backpropagate( ctx, inDelta );
}

BaseLayerCtx * BaseLayer :: createCtx( const DataVector * input ) const
{
	BaseLayerCtx * ctx = newCtx( input );

	ctx->getForwardCtx().getOutput().resize( getOutputSize() );
	ctx->getBackwardCtx().getDelta().resize( getOutputSize() );

	return ctx;
}

const size_t BaseLayer :: getInputSize() const
{
	return gx_dims_flatten_size( mInputDims );
}

const Dims & BaseLayer :: getInputDims() const
{
	return mInputDims;
}

const size_t BaseLayer :: getOutputSize() const
{
	return gx_dims_flatten_size( mOutputDims );
}

const Dims & BaseLayer :: getOutputDims() const
{
	return mOutputDims;
}

int BaseLayer :: getType() const
{
	return mType;
}

void BaseLayer :: setActFunc( ActFunc * actFunc )
{
	if( NULL != mActFunc ) delete mActFunc;
	mActFunc = actFunc;
}

const ActFunc * BaseLayer :: getActFunc() const
{
	return mActFunc;
}

void BaseLayer :: setTraining( bool isTraining )
{
	mIsTraining = isTraining;
}

void BaseLayer :: print( bool isDetail ) const
{
	printf( "Type = %d; ActFuncType = %d; InputDims = %s; OutputDims = %s;\n",
			mType, mActFunc ? mActFunc->getType() : -1,
			gx_vector2string( mInputDims ).c_str(),
			gx_vector2string( mOutputDims ).c_str() );

	printWeights( isDetail );
}

void BaseLayer :: collectGradients( BaseLayerCtx * ctx ) const
{
	/* do nothing */
}

void BaseLayer :: applyGradients( BackwardCtx * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	/* do nothing */
}

////////////////////////////////////////////////////////////

FullConnLayer :: FullConnLayer( const Dims & inputDims, size_t neuronCount )
	: BaseLayer( BaseLayer::eFullConn )
{
	mWeights.resize( neuronCount );
	for( auto & neuron : mWeights ) {
		neuron.resize( gx_dims_flatten_size( inputDims ) );
		for( auto & w : neuron ) w = gx_is_inner_debug ? 0.5 : Utils::random();
	}

	mBiases.resize( neuronCount );
	for( auto & b : mBiases ) b = gx_is_inner_debug ? 0.5 : Utils::random();

	mInputDims = inputDims;
	mOutputDims = { neuronCount };
}

FullConnLayer :: ~FullConnLayer()
{
}

void FullConnLayer :: printWeights( bool isDetail ) const
{
	if( !isDetail ) return;

	printf( "Weights: Count = %zu; InputCount = %zu;\n", mWeights.size(), mWeights[ 0 ].size() );
	for( size_t i = 0; i < mWeights.size() && i < 10; i++ ) {
		printf( "\tNeuron#%zu: WeightCount = %zu, Bias = %.8f\n", i, mWeights[ i ].size(), mBiases[ i ] );
		for( size_t j = 0; j < mWeights[ i ].size() && j < 10; j++ ) {
			printf( "\t\tWeight#%zu: %.8f\n", j, mWeights[ i ][ j ] );
		}

		if( mWeights[ i ].size() > 10 ) printf( "\t\t......\n" );
	}

	if( mWeights.size() > 10 ) printf( "\t......\n" );
}

const DataMatrix & FullConnLayer :: getWeights() const
{
	return mWeights;
}

const DataVector & FullConnLayer :: getBiases() const
{
	return mBiases;
}

void FullConnLayer :: setWeights( const DataMatrix & weights, const DataVector & biases )
{
	mWeights = weights;
	mBiases = biases;
}

void FullConnLayer :: calcOutput( BaseLayerCtx * ctx ) const
{
	assert( ctx->getForwardCtx().getOutput().size() == mWeights.size() );

	ForwardCtx & fwdCtx = ctx->getForwardCtx();

	for( size_t i = 0; i < mWeights.size(); i++ ) {
		//( *output )[ i ]  = ( mWeights[ i ] * input ).sum();
		fwdCtx.getOutput()[ i ]  = gx_inner_product( mWeights[ i ], fwdCtx.getInput() );
		if( ! gx_is_inner_debug )  fwdCtx.getOutput()[ i ] += mBiases[ i ];
	}

	if( gx_is_inner_debug ) {
		Utils::printVector( "input", fwdCtx.getInput() );
		Utils::printVector( "output", fwdCtx.getOutput() );
	}
}

void FullConnLayer :: backpropagate( BaseLayerCtx * ctx, DataVector * inDelta ) const
{
	BackwardCtx & bwdCtx = ctx->getBackwardCtx();

	DataVector temp( bwdCtx.getDelta().size() );

	if( NULL != inDelta ) {
		for( size_t i = 0; i < inDelta->size(); i++ ) {
			for( size_t j = 0; j < bwdCtx.getDelta().size(); j++ ) temp[ j ] = mWeights[ j ][ i ];
			( *inDelta )[ i ] = gx_inner_product( bwdCtx.getDelta(), temp );
		}
	}
}

BaseLayerCtx * FullConnLayer :: newCtx( const DataVector * input ) const
{
	BaseLayerCtx* ctx = new BaseLayerCtx( input );

	BackwardCtx & bwdCtx = ctx->getBackwardCtx();

	bwdCtx.getGradients().reserve( getOutputSize() );

	for( size_t i = 0; i < getOutputSize(); i++ ) {
		bwdCtx.getGradients().emplace_back( DataVector( getInputSize() ) );
	}

	return ctx;
}

void FullConnLayer :: collectGradients( BaseLayerCtx * ctx ) const
{
	ForwardCtx & fwdCtx = ctx->getForwardCtx();
	BackwardCtx & bwdCtx = ctx->getBackwardCtx();

	for( size_t i = 0; i < mWeights.size(); i++ ) {
		gx_vs_product( fwdCtx.getInput(), bwdCtx.getDelta()[ i ], &( bwdCtx.getGradients()[ i ] ) );
	}
}

void FullConnLayer :: applyGradients( BackwardCtx * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	for( size_t n = 0; n < mWeights.size(); n++ ) {
		optim->update( &( mWeights[ n ] ), ctx->getGradients()[ n ], trainingCount, miniBatchCount );
	}

	if( ! gx_is_inner_debug ) optim->updateBiases( &mBiases, ctx->getDelta(), miniBatchCount );
}

}; // namespace gxnet;

