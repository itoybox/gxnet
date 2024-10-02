#include "layer.h"
#include "context.h"

#include "utils.h"
#include "activation.h"
#include "optim.h"

#include <limits.h>
#include <cstdio>

#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>
#include <execution>

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

void BaseLayer :: forward( BaseLayerContext * ctx ) const
{
	calcOutput( ctx );

	if( NULL != mActFunc ) {
		MDSpanRO outMS( ctx->getOutMS() );

		if( gx_is_inner_debug ) Utils::printMDSpan( "before.act", outMS );

		mActFunc->activate( outMS, &( ctx->getOutMS() ) );

		if( gx_is_inner_debug ) Utils::printMDSpan( "after.act", ctx->getOutRO() );
	}

	ctx->getDelta().resize( ctx->getOutMS().data().size() );
}

void BaseLayer :: backward( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	if( NULL != mActFunc ) {
		MDSpanRO outMS( ctx->getOutMS() );
		MDSpanRW deltaMS( ctx->getDelta(), ctx->getOutMS().dims() );
		mActFunc->derivate( outMS, &deltaMS );
	}

	if( NULL != inDelta ) backpropagate( ctx, inDelta );
}

BaseLayerContext * BaseLayer :: createCtx() const
{
	return newCtx();
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
	printf( "Type = %d; ActFuncType = %d; \n",
			mType, mActFunc ? mActFunc->getType() : -1 );

	printWeights( isDetail );
}

void BaseLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	/* do nothing */
}

void BaseLayer :: applyGradients( BackwardContext * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	/* do nothing */
}

////////////////////////////////////////////////////////////

FullConnLayer :: FullConnLayer( size_t neuronCount, size_t inSize )
	: BaseLayer( eFullConn )
{
	mWeights.resize( neuronCount );
	for( auto & neuron : mWeights ) {
		neuron.resize( inSize );
		for( auto & w : neuron ) w = gx_is_inner_debug ? 0.5 : Utils::random();
	}

	mBiases.resize( neuronCount );
	for( auto & b : mBiases ) b = gx_is_inner_debug ? 0.5 : Utils::random();
}

FullConnLayer :: ~FullConnLayer()
{
}

void FullConnLayer :: printWeights( bool isDetail ) const
{
	if( !isDetail ) return;

	printf( "Weights: Count = %zu; InSize = %zu;\n", mWeights.size(), mWeights[ 0 ].size() );
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

void FullConnLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	size_t total = gx_dims_flatten_size( ctx->getInMS().dims() );

	size_t inSize = mWeights[ 0 ].size();
	size_t sampleCount = total / inSize;

	MDSpanRW & outMS = ctx->getOutMS();

	outMS.dims() = { sampleCount, mWeights.size() };
	outMS.data().resize( gx_dims_flatten_size( ctx->getOutMS().dims() ) );

	const DataType * input = std::begin( ctx->getInMS().data() );

	for( size_t n = 0; n < sampleCount; n++, input += inSize ) {
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			outMS( n, i ) = gx_inner_product( std::begin( mWeights[ i ] ), input, inSize );
			if( !gx_is_inner_debug )  outMS( n, i ) += mBiases[ i ];
		}
	}

	if( gx_is_inner_debug ) {
		Utils::printMDSpan( "input", ctx->getInMS() );
		Utils::printMDSpan( "output", ctx->getOutRO() );
	}
}

void FullConnLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	if( NULL == inDelta ) return;

	FullConnLayerContext * impl = dynamic_cast< FullConnLayerContext * >( ctx );

	size_t total = gx_dims_flatten_size( ctx->getOutMS().dims() );
	size_t sampleCount = total / mWeights.size();

	Dims inDeltaDims = { sampleCount, mWeights[ 0 ].size() };

	inDelta->resize( gx_dims_flatten_size( inDeltaDims ) );

	MDSpanRW inDeltaMS( *inDelta, inDeltaDims );

	DataVector & temp = impl->getTempWeights();
	temp.resize( mWeights.size() );

	for( size_t i = 0; i < mWeights[ 0 ].size(); i++ ) {
		for( size_t j = 0; j < mWeights.size(); j++ ) temp[ j ] = mWeights[ j ][ i ];

		const DataType * delta = std::begin( ctx->getDelta() );
		for( size_t n = 0; n < sampleCount; n++, delta += temp.size() ) {
			inDeltaMS( n, i ) = gx_inner_product( delta, std::begin( temp ), temp.size() );
		}
	}
}

BaseLayerContext * FullConnLayer :: newCtx() const
{
	return new FullConnLayerContext();
}

void FullConnLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	size_t total = gx_dims_flatten_size( ctx->getInMS().dims() );

	size_t inSize = mWeights[ 0 ].size();
	size_t sampleCount = total / inSize;

	if( ctx->getGradients().size() <= 0 ) {
		ctx->getGradients().reserve( mWeights.size() );
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			ctx->getGradients().emplace_back( DataVector( inSize ) );
		}
	}

	const DataType * input = std::begin( ctx->getInMS().data() );
	const DataType * delta = std::begin( ctx->getDelta() );

	FullConnLayerContext * impl = dynamic_cast< FullConnLayerContext * >( ctx );
	DataVector & tempGradients = impl->getTempGradients();
	tempGradients.resize( inSize );

	//DataVector tempGradients( inSize );

	for( size_t n = 0; n < sampleCount; n++, input += inSize, delta += mWeights.size() ) {
		for( size_t i = 0; i < mWeights.size(); i++ ) {
			gx_vs_product( input,  delta[ i ], std::begin( tempGradients ), inSize );

			if( n == 0 ) {
				//ctx->getGradients()[ i ] = tempGradients;
				std::copy( std::begin( tempGradients ), std::end( tempGradients ), std::begin( ctx->getGradients()[ i ] ) );
			} else {
				//ctx->getGradients()[ i ] += tempGradients;
				std::transform( std::begin( tempGradients ), std::end( tempGradients ),
						std::begin( ctx->getGradients()[ i ] ), std::begin( ctx->getGradients()[ i ] ),
						std::plus< DataType >() );
			}
		}
	}
}

void FullConnLayer :: applyGradients( BackwardContext * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	for( size_t n = 0; n < mWeights.size(); n++ ) {
		optim->update( &( mWeights[ n ] ), ctx->getGradients()[ n ], trainingCount, miniBatchCount );
	}

	if( !gx_is_inner_debug ) optim->updateBiases( &mBiases, ctx->getDelta(), miniBatchCount );
}

}; // namespace gxnet;

