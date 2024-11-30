#include "layer.h"
#include "context.h"

#include "utils.h"
#include "activation.h"
#include "optim.h"

#include "im2rows.h"

#include <limits.h>
#include <cstdio>

#include <iostream>
#include <numeric>
#include <algorithm>
#include <memory>

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
		if( gx_is_inner_debug ) Utils::printMDVector( "before.act", ctx->getOutput() );

		mActFunc->activate( ctx->getOutput(), &( ctx->getOutput() ) );

		if( gx_is_inner_debug ) Utils::printMDVector( "after.act", ctx->getOutput() );
	}

	ctx->getDelta().second = ctx->getOutput().second;
	ctx->getDelta().first.resize( ctx->getOutput().first.size() );
}

void BaseLayer :: backward( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	if( NULL != mActFunc ) {
		mActFunc->derivate( ctx->getOutput(), &( ctx->getDelta() ) );
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

const Dims & BaseLayer :: getBaseInDims() const
{
	return mBaseInDims;
}

size_t BaseLayer :: getBaseInSize() const
{
	return gx_dims_flatten_size( mBaseInDims );
}

const Dims & BaseLayer :: getBaseOutDims() const
{
	return mBaseOutDims;
}

size_t BaseLayer :: getBaseOutSize() const
{
	return gx_dims_flatten_size( mBaseOutDims );
}

void BaseLayer :: print( bool isDetail ) const
{
	printf( "Type = %d; ActFuncType = %d; BaseInDims = %s; BaseOutDims = %s; \n",
			mType, mActFunc ? mActFunc->getType() : -1,
			gx_vector2string( mBaseInDims ).c_str(),
			gx_vector2string( mBaseOutDims ).c_str() );

	printWeights( isDetail );
}

void BaseLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	/* do nothing */
}

void BaseLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	/* do nothing */
}

////////////////////////////////////////////////////////////

FullConnLayer :: FullConnLayer( const Dims & baseInDims, size_t neuronCount )
	: BaseLayer( eFullConn )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { neuronCount };

	mWeights.second = { neuronCount, gx_dims_flatten_size( mBaseInDims ) };
	mWeights.first.resize( gx_dims_flatten_size( mWeights.second ) );
	for( auto & item : mWeights.first ) {
		item = gx_is_inner_debug ? gx_debug_weight : Utils::random();
	}

	mBiases.resize( neuronCount );
	for( auto & b : mBiases ) b = gx_is_inner_debug ? gx_debug_weight : Utils::random();
}

FullConnLayer :: ~FullConnLayer()
{
}

void FullConnLayer :: printWeights( bool isDetail ) const
{
	if( !isDetail ) return;

	printf( "Weights: Count = %zu; InSize = %zu;\n", mWeights.second[ 0 ], mWeights.second[ 1 ] );
	MDSpanRO weightsRO( mWeights );
	for( size_t i = 0; i < weightsRO.dim( 0 ) && i < 10; i++ ) {
		printf( "\tNeuron#%zu: WeightCount = %zu, Bias = %.8f\n", i, weightsRO.dim( 0 ), mBiases[ i ] );
		for( size_t j = 0; j < weightsRO.dim( 1 ) && j < 10; j++ ) {
			printf( "\t\tWeight#%zu: %.8f\n", j, weightsRO( i, j ) );
		}

		if( weightsRO.dim( 1 ) > 10 ) printf( "\t\t......\n" );
	}

	if( weightsRO.dim( 0 ) > 10 ) printf( "\t......\n" );
}

const MDVector & FullConnLayer :: getWeights() const
{
	return mWeights;
}

const DataVector & FullConnLayer :: getBiases() const
{
	return mBiases;
}

void FullConnLayer :: setWeights( const MDVector & weights, const DataVector & biases )
{
	mWeights = weights;
	mBiases = biases;
}

void FullConnLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	MDSpanRO weightsRO( mWeights );

	const MDVector & inMD = ctx->getInput();
	MDVector & outMD = ctx->getOutput();

	size_t total = gx_dims_flatten_size( inMD.second );
	size_t inSize = gx_dims_flatten_size( mBaseInDims );
	size_t sampleCount = total / inSize;

	outMD.second = { sampleCount, weightsRO.dim( 0 ) };
	outMD.first.resize( gx_dims_flatten_size( outMD.second ) );

	Dims inDims = { sampleCount, inSize };
	MDSpanRO inRO( std::begin( inMD.first ), inDims );

	if( gx_is_inner_debug ) {
		gx_rows_product( inRO, weightsRO, std::begin( outMD.first ), outMD.first.size() );
	} else {
		gx_rows_product( inRO, weightsRO, mBiases, false, std::begin( outMD.first ), outMD.first.size() );
	}

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "input", inMD );
		Utils::printMDVector( "output", outMD );
	}
}

void FullConnLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	if( NULL == inDelta ) return;

	MDSpanRO weightsRO( mWeights );

	MDSpanRO deltaRO( ctx->getDelta() );

	gx_matmul( deltaRO, weightsRO, inDelta );
}

BaseLayerContext * FullConnLayer :: newCtx() const
{
	return new FullConnLayerContext();
}

void FullConnLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	FullConnLayerContext * ctxImpl = dynamic_cast< FullConnLayerContext * >( ctx );

	MDSpanRO weightsRO( mWeights );

	const MDVector & inMD = ctx->getInput();
	const MDVector & deltaMD = ctx->getDelta();

	size_t total = gx_dims_flatten_size( inMD.second );
	size_t inSize = gx_dims_flatten_size( mBaseInDims );
	size_t sampleCount = total / inSize;

	MDVector & gradients = ctx->getGradients();
	DataVector & tempGradients = ctxImpl->getTempGradients();

	if( gradients.first.size() <= 0 ) {
		gradients.second = mWeights.second;
		gradients.first.resize( mWeights.first.size() );
		tempGradients.resize( mWeights.first.size() );
	}

	const DataType * input = std::begin( inMD.first );
	const DataType * delta = std::begin( deltaMD.first );

	for( size_t n = 0; n < sampleCount; n++, input += inSize, delta += weightsRO.dim( 0 ) ) {

#if 0
		if( 0 == n ) {
			gx_kronecker_product( delta, weightsRO.dim( 0 ), input, inSize,
					std::begin( gradients.first ), gradients.first.size() );
		} else {
			gx_kronecker_product( delta, weightsRO.dim( 0 ), input, inSize,
					std::begin( tempGradients ), tempGradients.size() );
			std::transform( std::begin( tempGradients ), std::end( tempGradients ),
					std::begin( gradients.first ), std::begin( gradients.first ),
					std::plus< DataType >() );
		}
#else
		for( size_t i = 0; i < weightsRO.dim( 0 ); i++ ) {
			if( n == 0 ) {
				gx_vs_product( input, delta[ i ], std::begin( gradients.first ) + i * inSize, inSize );
			} else {
				gx_vs_product( input, delta[ i ], std::begin( tempGradients ), inSize );
				std::transform( std::begin( tempGradients ), std::begin( tempGradients ) + inSize,
						std::begin( gradients.first ) + i * inSize, std::begin( gradients.first ) + i * inSize,
						std::plus< DataType >() );
			}
		}
#endif

	}
}

void FullConnLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	optim->update( &( mWeights.first ), ctx.getGradients().first, trainingCount, miniBatchCount );

	if( !gx_is_inner_debug ) optim->updateBiases( &mBiases, ctx.getDelta().first, miniBatchCount );
}

////////////////////////////////////////////////////////////

ConvLayer :: ConvLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize )
	: BaseLayer( BaseLayer::eConv )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = {
		filterCount,
		mBaseInDims[ 1 ] - filterSize + 1,
		mBaseInDims[ 2 ] - filterSize + 1
	};

	mFilters.second = { filterCount, mBaseInDims[ 0 ], filterSize, filterSize };

	mFilters.first.resize( gx_dims_flatten_size( mFilters.second) );
	for( auto & item : mFilters.first ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();

	mBiases.resize( filterCount );
	for( auto & item : mBiases ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();
}

ConvLayer :: ConvLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases )
	: BaseLayer( BaseLayer::eConv )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = {
		filters.second[ 0 ],
		mBaseInDims[ 1 ] - filters.second[ 2 ] + 1,
		mBaseInDims[ 2 ] - filters.second[ 3 ] + 1
	};

	mFilters = filters;

	mBiases = biases;
}

ConvLayer :: ~ConvLayer()
{
}

BaseLayerContext * ConvLayer :: newCtx() const
{
	ConvLayerContext * ctx = new ConvLayerContext();

	return ctx;
}

void ConvLayer :: printWeights( bool isDetail ) const
{
	printf( "\nfilterDims = %s\n", gx_vector2string( mFilters.second ).c_str() );

	if( !isDetail ) return;

	Utils::printMDVector( "filters", mFilters );
	Utils::printVector( "biases", mBiases );
}

const MDVector & ConvLayer :: getFilters() const
{
	return mFilters;
}

const DataVector & ConvLayer :: getBiases() const
{
	return mBiases;
}

void ConvLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInput().second;

	assert( inDims.size() == 4 );

	Dims & outDims = ctx->getOutput().second;
	outDims = {
		inDims[ 0 ], mFilters.second[ 0 ],
		inDims[ 2 ] - mFilters.second[ 2 ] + 1,
		inDims[ 3 ] - mFilters.second[ 3 ] + 1
	};
	ctx->getOutput().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outRW( ctx->getOutput() );

	MDSpanRO filterRO( mFilters );

	MDSpanRO inRO( ctx->getInput() );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < filterRO.dim( 0 ); f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outRW( n, f, x, y ) = forwardConv( inRO, n, f, x, y, filterRO ) + mBiases[ f ];
				}
			}
		}
	}
}

DataType ConvLayer :: forwardConv( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterRO )
{
	DataType total = 0;

	for( size_t c = 0; c < filterRO.dim( 1 ); c++ ) {
		for( size_t x = 0; x < filterRO.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterRO.dim( 3 ); y++ ) {
				total += inRO( sampleIndex, c, beginX + x, beginY + y ) * filterRO( filterIndex, c, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	ConvLayerContext * ctxImpl = dynamic_cast< ConvLayerContext * >( ctx );

	const Dims & outDims = ctx->getOutput().second;

	// 1. prepare outDelta padding data
	MDVector & paddingDelta = ctxImpl->getPaddingDelta();
	if( paddingDelta.second.size() <= 0 ) {
		paddingDelta.second = {
				outDims[ 0 ],
				outDims[ 1 ],
				outDims[ 2 ] + 2 * ( mFilters.second[ 2 ] - 1 ),
				outDims[ 3 ] + 2 * ( mFilters.second[ 3 ] - 1 )
		};
	}
	paddingDelta.second[ 0 ] = outDims[ 0 ];

	paddingDelta.first.resize( gx_dims_flatten_size( paddingDelta.second ) );

	MDSpanRW paddingDeltaRW( paddingDelta );

	MDSpanRO deltaRO( ctx->getDelta() );
	copyOutDelta( deltaRO, mFilters.second[ 2 ], &paddingDeltaRW );

	if( gx_is_inner_debug ) Utils::printMDVector( "paddingDelta", paddingDelta );

	// 2. prepare rotate180 filters
	MDVector rot180Filters;
	Im2Rows::rot180Filters( mFilters, &rot180Filters );
	if( gx_is_inner_debug ) Utils::printMDVector( "rot180Filters", rot180Filters );

	// 3. convolution
	MDSpanRO rot180FiltersRO( rot180Filters );
	MDSpanRO paddingDeltaRO( paddingDelta );

	MDSpanRW inDeltaRW( *inDelta );

	for( size_t n = 0; n < inDeltaRW.dim( 0 ); n++ ) {
		for( size_t c = 0; c < inDeltaRW.dim( 1 ); c++ ) {
			for( size_t x = 0; x < inDeltaRW.dim( 2 ); x++ ) {
				for( size_t y = 0; y < inDeltaRW.dim( 3 ); y++ ) {
					inDeltaRW( n, c, x, y ) = backwardConv( paddingDeltaRO, n, c, x, y, rot180FiltersRO );
				}
			}
		}
	}
}

DataType ConvLayer :: backwardConv( const MDSpanRO & inRO, size_t sampleIndex, size_t channelIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterRO )
{
	DataType total = 0;

	for( size_t f = 0; f < filterRO.dim( 0 ); f++ ) {
		for( size_t x = 0; x < filterRO.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterRO.dim( 3 ); y++ ) {
				total += inRO( sampleIndex, f, beginX + x, beginY + y ) * filterRO( f, channelIndex, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: copyOutDelta( const MDSpanRO & outDeltaRO, size_t filterSize, MDSpanRW * outPaddingRW )
{
	for( size_t n = 0; n < outDeltaRO.dim( 0 ); n++ ) {
		for( size_t f = 0; f < outDeltaRO.dim( 1 ); f++ ) {
			for( size_t x = 0; x < outDeltaRO.dim( 2 ); x++ ) {
				for( size_t y = 0; y < outDeltaRO.dim( 3 ); y++ ) {
					( *outPaddingRW )( n, f, x + filterSize - 1, y + filterSize - 1 ) = outDeltaRO( n, f, x, y );
				}
			}
		}
	}
}

void ConvLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	const Dims & outDims = ctx->getOutput().second;

	MDVector & gradients = ctx->getGradients();
	if( gradients.first.size() <= 0 ) {
		gradients.second = mFilters.second;
		gradients.first.resize( mFilters.first.size() );
	} else {
		gradients.first = 0.0;
	}

	MDSpanRW gradientRW( gradients );
	MDSpanRO deltaRO( ctx->getDelta() );

	const MDSpanRO inRO( ctx->getInput() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < mFilters.second[ 0 ]; f++ ) {
			for( size_t c = 0; c < mFilters.second[ 1 ]; c++ ) {
				for( size_t x = 0; x < mFilters.second[ 2 ]; x++ ) {
					for( size_t y = 0; y < mFilters.second[ 3 ]; y++ ) {
						gradientRW( f, c, x, y ) += gradientConv( inRO, n, f, c, x, y, deltaRO );
					}
				}
			}
		}
	}
}

DataType ConvLayer :: gradientConv( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
		size_t channelIndex, size_t beginX, size_t beginY, const MDSpanRO & filterRO )
{
	DataType total = 0;

	for( size_t x = 0; x < filterRO.dim( 2 ); x++ ) {
		for( size_t y = 0; y < filterRO.dim( 3 ); y++ ) {
			total += inRO( sampleIndex, channelIndex, beginX + x, beginY + y ) * filterRO( sampleIndex, filterIndex, x, y );
		}
	}

	return total;
}

void ConvLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	optim->update( &mFilters.first, ctx.getGradients().first, trainingCount, miniBatchCount );

	const Dims & deltaDims = ctx.getDelta().second;

	DataVector biasDelta( deltaDims[ 1 ] );

	const MDSpanRO deltaRO( ctx.getDelta() );

	for( size_t n = 0; n < deltaDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < deltaDims[ 1 ]; f++ ) {
			for( size_t i = 0; i < deltaDims[ 2 ]; i++ ) {
				for( size_t j = 0; j < deltaDims[ 3 ]; j++ ) {
					biasDelta[ f ] += deltaRO( n, f, i, j );
				}
			}
		}
	}

	if( gx_is_inner_debug ) Utils::printVector( "bias.delta", biasDelta );

	optim->updateBiases( &mBiases, biasDelta, miniBatchCount );
}

////////////////////////////////////////////////////////////

ConvExLayer :: ConvExLayer( const Dims & baseInDims, size_t filterCount, size_t filterSize )
	: ConvLayer( baseInDims, filterCount, filterSize )
{
	mType = eConvEx;

	Im2Rows::rot180Filters2Rows( mFilters, &mRowsOfRot180Filters );
}

ConvExLayer :: ConvExLayer( const Dims & baseInDims, const MDVector & filters, const DataVector & biases )
	: ConvLayer( baseInDims, filters, biases )
{
	mType = eConvEx;

	Im2Rows::rot180Filters2Rows( mFilters, &mRowsOfRot180Filters );
}

ConvExLayer :: ~ConvExLayer()
{
}

BaseLayerContext * ConvExLayer :: newCtx() const
{
	ConvExLayerContext * ctx = new ConvExLayerContext();

	return ctx;
}

void ConvExLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const Dims & inDims = ctx->getInput().second;

	Dims & outDims = ctx->getOutput().second;

	outDims = mBaseOutDims;
	outDims.insert( outDims.begin(), inDims[ 0 ] );

	ctx->getOutput().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRO inRO( ctx->getInput() );

	DataType * outPtr = std::begin( ctx->getOutput().first );
	size_t outSize = gx_dims_flatten_size( mBaseOutDims );

	Dims fakeDims = { mFilters.second[ 0 ],
			gx_dims_flatten_size( mFilters.second ) / mFilters.second[ 0 ] };
	MDSpanRO filterRO( std::begin( mFilters.first ), fakeDims );

	MDVector & rows4input = ctxImpl->getRows4calcOutput();

	for( size_t n = 0; n < inDims[ 0 ]; n++, outPtr += outSize ) {

		Im2Rows::input2Rows( inRO, n, mFilters.second, &rows4input );

		if( gx_is_inner_debug ) Utils::printMDVector( "input", rows4input );

		MDSpanRO inputRO( rows4input );
		gx_rows_product( filterRO, inputRO, mBiases, true, outPtr, outSize );
	}
}

void ConvExLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	const Dims & outDims = ctx->getOutput().second;

	MDVector & paddingDelta = ctxImpl->getPaddingDelta();

	// prepare outDelta padding data
	if( paddingDelta.second.size() <= 0 ) {
		paddingDelta.second = {
				outDims[ 0 ],
				outDims[ 1 ],
				outDims[ 2 ] + 2 * ( mFilters.second[ 2 ] - 1 ),
				outDims[ 3 ] + 2 * ( mFilters.second[ 3 ] - 1 )
		};
	}
	paddingDelta.second[ 0 ] = outDims[ 0 ];

	paddingDelta.first.resize( gx_dims_flatten_size( paddingDelta.second ) );

	MDSpanRW paddingDeltaRW( paddingDelta );
	MDSpanRO deltaRO( ctx->getDelta() );
	copyOutDelta( deltaRO, mFilters.second[ 2 ], &paddingDeltaRW );

	if( gx_is_inner_debug ) Utils::printMDVector( "outPadding", paddingDelta );

	if( gx_is_inner_debug ) Utils::printMDVector( "rot180filters", mRowsOfRot180Filters );

	MDSpanRO paddingDeltaRO( paddingDelta );

	// rot180Filters dims
	Dims fakeDims = { mFilters.second[ 1 ], mFilters.second[ 0 ], mFilters.second[ 2 ], mFilters.second[ 3 ] };

	MDVector & rows4delta = ctxImpl->getRows4backpropagate();

	size_t inDeltaSize = gx_dims_flatten_size( mBaseInDims );
	DataType * inDeltaPtr = std::begin( inDelta->first );

	MDSpanRO filterRO( mRowsOfRot180Filters );

	for( size_t n = 0; n < outDims[ 0 ]; n++, inDeltaPtr += inDeltaSize ) {
		Im2Rows::input2Rows( paddingDeltaRO, n, fakeDims, &rows4delta );

		//if( gx_is_inner_debug ) Utils::printMatrix( "outPadding", rows4delta );

		MDSpanRO deltaRO( rows4delta );
		gx_rows_product( filterRO, deltaRO, inDeltaPtr, inDeltaSize );
	}
}

void ConvExLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	MDVector & gradients = ctx->getGradients();
	if( gradients.first.size() <= 0 ) {
		gradients.second = mFilters.second;
		gradients.first.resize( mFilters.first.size() );
	}

	const MDSpanRO inRO( ctx->getInput() );

	DataVector & tempGradients = ctxImpl->getTempGradients();
	MDVector & rows4input = ctxImpl->getRows4collectGradients();

	const Dims & outDims = ctx->getOutput().second;
	Dims deltaDims = { outDims[ 1 ], outDims[ 2 ] * outDims[ 3 ] };
	size_t deltaSize = gx_dims_flatten_size( deltaDims );

	for( size_t n = 0; n < inRO.dim( 0 ); n++ ) {

		MDSpanRO deltaRO( std::begin( ctx->getDelta().first ) + deltaSize * n, deltaDims );

		if( gx_is_inner_debug ) Utils::printMDSpan( "deltas", deltaRO );

		Im2Rows::input2Rows4Gradients( inRO, n, outDims, &rows4input );

		if( gx_is_inner_debug ) Utils::printMDVector( "input", rows4input );

		MDSpanRO inputRO( rows4input );

		if( 0 == n ) {
			gx_rows_product( deltaRO, inputRO, std::begin( gradients.first ), gradients.first.size() );
		} else {
			tempGradients.resize( gradients.first.size() );
			gx_rows_product( deltaRO, inputRO, std::begin( tempGradients ), tempGradients.size() );
			gradients.first += tempGradients;
		}
	}
}

void ConvExLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	ConvLayer::applyGradients( ctx, optim, trainingCount, miniBatchCount );

	Im2Rows::rot180Filters2Rows( mFilters, &mRowsOfRot180Filters );
}

////////////////////////////////////////////////////////////

MaxPoolLayer :: MaxPoolLayer( const Dims & baseInDims, size_t poolSize )
	: BaseLayer( BaseLayer::eMaxPool )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { mBaseInDims[ 0 ], mBaseInDims[ 1 ] / poolSize, mBaseInDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

MaxPoolLayer :: ~MaxPoolLayer()
{
}

void MaxPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t MaxPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

BaseLayerContext * MaxPoolLayer :: newCtx() const
{
	BaseLayerContext * ctx = new BaseLayerContext();

	return ctx;
}

void MaxPoolLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInput().second;

	Dims & outDims = ctx->getOutput().second;

	outDims = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	ctx->getOutput().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outRW( ctx->getOutput() );
	MDSpanRO inRO( ctx->getInput() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outRW( n, f, x, y ) = pool( inRO, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType MaxPoolLayer :: pool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 1.0 * INT_MIN;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result = std::max( result, inRO( sampleIndex, filterIndex, beginX + x, beginY + y ) );
		}
	}

	return result;
}

void MaxPoolLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	const Dims & outDims = ctx->getOutput().second;

	MDSpanRW inDeltaRW( *inDelta );

	MDSpanRO outRO( ctx->getOutput() );

	MDSpanRO inRO( ctx->getInput() );
	MDSpanRO outDeltaRO( ctx->getDelta() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inRO, n, f, x * mPoolSize, y * mPoolSize,
							outRO( n, f, x, y ), outDeltaRO( n, f, x, y ), &inDeltaRW );
				}
			}
		}
	}
}

void MaxPoolLayer :: unpool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY,
		const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaRW ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			if( inRO( sampleIndex, filterIndex, beginX + x, beginY + y ) == maxValue ) {
				( *inDeltaRW )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta;
			}
		}
	}
}

////////////////////////////////////////////////////////////

AvgPoolLayer :: AvgPoolLayer( const Dims & baseInDims, size_t poolSize )
	: BaseLayer( BaseLayer::eAvgPool )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = { mBaseInDims[ 0 ], mBaseInDims[ 1 ] / poolSize, mBaseInDims[ 2 ] / poolSize };

	mPoolSize = poolSize;
}

AvgPoolLayer :: ~AvgPoolLayer()
{
}

void AvgPoolLayer :: printWeights( bool isDetail ) const
{
	printf( "\nPoolSize = %zu\n", mPoolSize );
}

size_t AvgPoolLayer :: getPoolSize() const
{
	return mPoolSize;
}

BaseLayerContext * AvgPoolLayer :: newCtx() const
{
	BaseLayerContext * ctx = new BaseLayerContext();

	return ctx;
}

void AvgPoolLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const Dims & inDims = ctx->getInput().second;

	Dims & outDims = ctx->getOutput().second;

	outDims = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	ctx->getOutput().first.resize( gx_dims_flatten_size( outDims ) );

	MDSpanRW outRW( ctx->getOutput() );
	MDSpanRO inRO( ctx->getInput() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outRW( n, f, x, y ) = pool( inRO, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType AvgPoolLayer :: pool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 0;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result += inRO( sampleIndex, filterIndex, beginX + x, beginY + y );
		}
	}

	return result / ( mPoolSize * mPoolSize );
}

void AvgPoolLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	const Dims & outDims = ctx->getOutput().second;

	MDSpanRW inDeltaRW( *inDelta );

	MDSpanRO outRO( ctx->getOutput() );

	MDSpanRO inRO( ctx->getInput() );
	MDSpanRO outDeltaRO( ctx->getDelta() );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inRO, n, f, x * mPoolSize, y * mPoolSize,
							outRO( n, f, x, y ), outDeltaRO( n, f, x, y ), &inDeltaRW );
				}
			}
		}
	}
}

void AvgPoolLayer :: unpool( const MDSpanRO & inRO, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaRW ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			( *inDeltaRW )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta / ( mPoolSize * mPoolSize );
		}
	}
}

////////////////////////////////////////////////////////////

DropoutLayer :: DropoutLayer( const Dims & baseInDims, DataType dropRate )
	: BaseLayer( eDropout )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = baseInDims;

	mDropRate = dropRate;
	mIsTraining = false;
}

DropoutLayer :: ~DropoutLayer()
{
}

void DropoutLayer :: printWeights( bool isDetail ) const
{
	printf( "\nDropRate = %f\n", mDropRate );
}

DataType DropoutLayer :: getDropRate() const
{
	return mDropRate;
}

BaseLayerContext * DropoutLayer :: newCtx() const
{
	DropoutLayerContext * ctx = new DropoutLayerContext();

	return ctx;
}

void DropoutLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	DropoutLayerContext * ctxImpl = dynamic_cast< DropoutLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const DataVector & input = ctx->getInput().first;
	DataVector & output = ctx->getOutput().first;

	ctx->getOutput().second = ctx->getInput().second;
	ctx->getOutput().first.resize( input.size() );

	BoolVector & mask = ctxImpl->getMask();
	mask.resize( input.size() );

	if( mIsTraining ) {
		for( size_t i = 0; i < input.size(); i++ ) {
			if( Utils::random( 0, 1 ) < mDropRate ) {
				mask[ i ] = true;
				output[ i ] = 0;
			} else {
				mask[ i ] = false;
				output[ i ] = input[ i ] / ( 1.0 - mDropRate );
			}
		}
	} else {
		output = input;
	}

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "dropout.input", ctx->getInput() );
		Utils::printMDVector( "dropout.output", ctx->getOutput() );
	}
}

void DropoutLayer :: backpropagate( BaseLayerContext * ctx, MDVector * inDelta ) const
{
	DropoutLayerContext * ctxImpl = dynamic_cast< DropoutLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const DataVector & delta = ctx->getDelta().first;
	BoolVector & mask = ctxImpl->getMask();

	for( size_t i = 0; i < delta.size(); i++ )
			inDelta->first[ i ] = mask[ i ] ? 0 : delta[ i ];

	if( gx_is_inner_debug ) {
		Utils::printMDVector( "dropout.outDelta", ctx->getDelta() );
		Utils::printMDVector( "dropout.inDelta", *inDelta );
	}
}


}; // namespace gxnet;

