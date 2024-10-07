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
		if( gx_is_inner_debug ) Utils::printMDSpan( "before.act", ctx->getOutRO() );

		mActFunc->activate( ctx->getOutRO(), &( ctx->getOutMS() ) );

		if( gx_is_inner_debug ) Utils::printMDSpan( "after.act", ctx->getOutRO() );
	}

	ctx->getDeltaMS().dims() = ctx->getOutMS().dims();
	ctx->getDeltaMS().data().resize( ctx->getOutMS().data().size() );
}

void BaseLayer :: backward( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	if( NULL != mActFunc ) {
		ctx->getDeltaMS().dims() = ctx->getOutMS().dims();

		mActFunc->derivate( ctx->getOutRO(), &( ctx->getDeltaMS() ) );
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

	mWeights.resize( neuronCount );
	for( auto & neuron : mWeights ) {
		neuron.resize( gx_dims_flatten_size( mBaseInDims ) );
		for( auto & w : neuron ) w = gx_is_inner_debug ? gx_debug_weight : Utils::random();
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

	FullConnLayerContext * ctxImpl = dynamic_cast< FullConnLayerContext * >( ctx );

	size_t total = gx_dims_flatten_size( ctx->getOutMS().dims() );
	size_t sampleCount = total / mWeights.size();

	Dims inDeltaDims = { sampleCount, mWeights[ 0 ].size() };

	inDelta->resize( gx_dims_flatten_size( inDeltaDims ) );

	MDSpanRW inDeltaMS( *inDelta, inDeltaDims );

	DataVector & temp = ctxImpl->getTempWeights();
	temp.resize( mWeights.size() );

	for( size_t i = 0; i < mWeights[ 0 ].size(); i++ ) {
		for( size_t j = 0; j < mWeights.size(); j++ ) temp[ j ] = mWeights[ j ][ i ];

		const DataType * delta = std::begin( ctx->getDeltaMS().data() );
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
	const DataType * delta = std::begin( ctx->getDeltaMS().data() );

	FullConnLayerContext * ctxImpl = dynamic_cast< FullConnLayerContext * >( ctx );
	DataVector & tempGradients = ctxImpl->getTempGradients();
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

void FullConnLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	for( size_t n = 0; n < mWeights.size(); n++ ) {
		optim->update( &( mWeights[ n ] ), ctx.getGradients()[ n ], trainingCount, miniBatchCount );
	}

	if( !gx_is_inner_debug ) optim->updateBiases( &mBiases, ctx.getDeltaRO().data(), miniBatchCount );
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

	mFilterDims = { filterCount, mBaseInDims[ 0 ], filterSize, filterSize };

	mFilters.resize( gx_dims_flatten_size( mFilterDims ) );
	for( auto & item : mFilters ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();

	mBiases.resize( filterCount );
	for( auto & item : mBiases ) item = gx_is_inner_debug ? gx_debug_weight : Utils::random();
}

ConvLayer :: ConvLayer( const Dims & baseInDims, const DataVector & filters,
		const Dims & filterDims, const DataVector & biases )
	: BaseLayer( BaseLayer::eConv )
{
	mBaseInDims = baseInDims;
	mBaseOutDims = {
		filterDims[ 0 ],
		mBaseInDims[ 1 ] - filterDims[ 2 ] + 1,
		mBaseInDims[ 2 ] - filterDims[ 3 ] + 1
	};

	mFilters = filters;

	mFilterDims = filterDims;

	mBiases = biases;
}

ConvLayer :: ~ConvLayer()
{
}

void ConvLayer :: printWeights( bool isDetail ) const
{
	printf( "\nfilterDims = %s\n", gx_vector2string( mFilterDims ).c_str() );

	if( !isDetail ) return;

	Utils::printVector( "filters", mFilters, mFilterDims );
	Utils::printVector( "biases", mBiases );
}

const Dims & ConvLayer :: getFilterDims() const
{
	return mFilterDims;
}

const DataVector & ConvLayer :: getFilters() const
{
	return mFilters;
}

const DataVector & ConvLayer :: getBiases() const
{
	return mBiases;
}

void ConvLayer :: calcOutput( BaseLayerContext * ctx ) const
{
	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	assert( inDims.size() == 4 );

	MDSpanRW & outMS = ctx->getOutMS();
	Dims & outDims = outMS.dims();
	outDims = {
		inDims[ 0 ], mFilterDims[ 0 ],
		inDims[ 2 ] - mFilterDims[ 2 ] + 1,
		inDims[ 3 ] - mFilterDims[ 3 ] + 1
	};
	outMS.data().resize( gx_dims_flatten_size( outDims ) );

	MDSpanRO filterMS( mFilters, mFilterDims );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < mFilterDims[ 0 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = forwardConv( inMS, n, f, x, y, filterMS ) + mBiases[ f ];
				}
			}
		}
	}
}

DataType ConvLayer :: forwardConv( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t c = 0; c < filterMS.dim( 1 ); c++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( sampleIndex, c, beginX + x, beginY + y ) * filterMS( filterIndex, c, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	MDSpanRW & outMS = ctx->getOutMS();
	Dims & outDims = outMS.dims();

	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	// 1. prepare outDelta padding data
	Dims outPaddingDims = {
			outDims[ 0 ],
			outDims[ 1 ],
			outDims[ 2 ] + 2 * ( mFilterDims[ 2 ] - 1 ),
			outDims[ 3 ] + 2 * ( mFilterDims[ 3 ] - 1 )
	};

	DataVector outPadding( gx_dims_flatten_size( outPaddingDims ) );

	MDSpanRW outPaddingMS( outPadding, outPaddingDims );
	copyOutDelta( ctx->getDeltaRO(), mFilterDims[ 2 ], &outPaddingMS );
	if( gx_is_inner_debug ) Utils::printVector( "outPadding", outPadding, outPaddingDims );

	// 2. prepare rotate180 filters
	DataVector rot180Filters( mFilters.size() );
	Im2Rows::rot180Filters( mFilters, mFilterDims, &rot180Filters );
	if( gx_is_inner_debug ) Utils::printVector( "rot180Filters", rot180Filters, mFilterDims );

	// 3. convolution
	MDSpanRO rot180FiltersMS( rot180Filters, mFilterDims );
	MDSpanRO outPaddingRO( outPadding, outPaddingDims );

	MDSpanRW inDeltaMS( *inDelta, inDims );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {
		for( size_t c = 0; c < inDims[ 1 ]; c++ ) {
			for( size_t x = 0; x < inDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < inDims[ 3 ]; y++ ) {
					inDeltaMS( n, c, x, y ) = backwardConv( outPaddingRO, n, c, x, y, rot180FiltersMS );
				}
			}
		}
	}
}

DataType ConvLayer :: backwardConv( const MDSpanRO & inMS, size_t sampleIndex, size_t channelIndex,
		size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t f = 0; f < filterMS.dim( 0 ); f++ ) {
		for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
			for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
				total += inMS( sampleIndex, f, beginX + x, beginY + y ) * filterMS( f, channelIndex, x, y );
			}
		}
	}

	return total;
}

void ConvLayer :: copyOutDelta( const MDSpanRO & outDeltaMS, size_t filterSize, MDSpanRW * outPaddingMS )
{
	for( size_t n = 0; n < outDeltaMS.dim( 0 ); n++ ) {
		for( size_t f = 0; f < outDeltaMS.dim( 1 ); f++ ) {
			for( size_t x = 0; x < outDeltaMS.dim( 2 ); x++ ) {
				for( size_t y = 0; y < outDeltaMS.dim( 3 ); y++ ) {
					( *outPaddingMS )( n, f, x + filterSize - 1, y + filterSize - 1 ) = outDeltaMS( n, f, x, y );
				}
			}
		}
	}
}

BaseLayerContext * ConvLayer :: newCtx() const
{
	BaseLayerContext * ctx = new BaseLayerContext();

	return ctx;
}

void ConvLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	MDSpanRW & outMS = ctx->getOutMS();
	Dims & outDims = outMS.dims();

	const MDSpanRO & inMS = ctx->getInMS();

	if( ctx->getGradients().size() <= 0 ) {
		ctx->getGradients().emplace_back( DataVector( gx_dims_flatten_size( mFilterDims ) ) );
	} else {
		ctx->getGradients()[ 0 ] = 0.0;
	}

	MDSpanRW gradientMS( ctx->getGradients()[ 0 ], mFilterDims );

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < mFilterDims[ 0 ]; f++ ) {
			for( size_t c = 0; c < mFilterDims[ 1 ]; c++ ) {
				for( size_t x = 0; x < mFilterDims[ 2 ]; x++ ) {
					for( size_t y = 0; y < mFilterDims[ 3 ]; y++ ) {
						gradientMS( f, c, x, y ) += gradientConv( inMS, n, f, c, x, y, ctx->getDeltaRO() );
					}
				}
			}
		}
	}
}

DataType ConvLayer :: gradientConv( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t channelIndex, size_t beginX, size_t beginY, const MDSpanRO & filterMS )
{
	DataType total = 0;

	for( size_t x = 0; x < filterMS.dim( 2 ); x++ ) {
		for( size_t y = 0; y < filterMS.dim( 3 ); y++ ) {
			total += inMS( sampleIndex, channelIndex, beginX + x, beginY + y ) * filterMS( sampleIndex, filterIndex, x, y );
		}
	}

	return total;
}

void ConvLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	optim->update( &mFilters, ctx.getGradients()[ 0 ], trainingCount, miniBatchCount );

	const MDSpanRO & deltaMS = ctx.getDeltaRO();
	const Dims & deltaDims = deltaMS.dims();

	DataVector biasDelta( deltaDims[ 1 ] );

	for( size_t n = 0; n < deltaDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < deltaDims[ 1 ]; f++ ) {
			for( size_t i = 0; i < deltaDims[ 2 ]; i++ ) {
				for( size_t j = 0; j < deltaDims[ 3 ]; j++ ) {
					biasDelta[ f ] += deltaMS( n, f, i, j );
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

	updateFiltersRows( mFilters, mFilterDims, &mRowsOfFilters, &mRowsOfRot180Filters );
}

ConvExLayer :: ConvExLayer( const Dims & baseInDims, const DataVector & filters, const Dims & filterDims,
		const DataVector & biases )
	: ConvLayer( baseInDims, filters, filterDims, biases )
{
	mType = eConvEx;

	updateFiltersRows( mFilters, mFilterDims, &mRowsOfFilters, &mRowsOfRot180Filters );
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

	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	MDSpanRW & outMS = ctx->getOutMS();
	Dims & outDims = outMS.dims();

	outDims = mBaseOutDims;
	outDims.insert( outDims.begin(), inDims[ 0 ] );

	outMS.data().resize( gx_dims_flatten_size( outDims ) );

	Dims outDims4Rows = { outDims[ 0 ], outDims[ 1 ], outDims[ 2 ] * outDims[ 3 ] };
	MDSpanRW outMS4Rows( outMS.data(), outDims4Rows );

	for( size_t n = 0; n < inDims[ 0 ]; n++ ) {

		Im2Rows::input2Rows( inMS, n, mFilterDims, &( ctxImpl->getRows4calcOutput() ) );

		if( gx_is_inner_debug ) Utils::printMatrix( "input", ctxImpl->getRows4calcOutput() );

		assert( ctx->getOutMS().data().size() == ( inDims[ 0 ] * ctxImpl->getRows4calcOutput().size() * mRowsOfFilters.size() ) );

		for( size_t i = 0; i < mRowsOfFilters.size(); i++ ) {
			for( size_t j = 0; j < ctxImpl->getRows4calcOutput().size(); j++ ) {
				outMS4Rows( n, i, j ) = gx_inner_product( std::begin( mRowsOfFilters[ i ] ),
						std::begin( ctxImpl->getRows4calcOutput()[ j ] ), mRowsOfFilters[ i ].size() );
				outMS4Rows( n, i, j ) += mBiases[ i ];
			}
		}
	}
}

void ConvExLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	MDSpanRW & outMS = ctx->getOutMS();
	Dims & outDims = outMS.dims();

	// prepare outDelta padding data
	Dims paddingDeltaDims = {
			outDims[ 0 ],
			outDims[ 1 ],
			outDims[ 2 ] + 2 * ( mFilterDims[ 2 ] - 1 ),
			outDims[ 3 ] + 2 * ( mFilterDims[ 3 ] - 1 )
	};

	ctxImpl->getPaddingDelta().resize( gx_dims_flatten_size( paddingDeltaDims ) );

	MDSpanRW paddingDeltaMS( ctxImpl->getPaddingDelta(), paddingDeltaDims );
	copyOutDelta( ctx->getDeltaRO(), mFilterDims[ 2 ], &paddingDeltaMS );

	if( gx_is_inner_debug ) Utils::printMatrix( "rot180filters", mRowsOfRot180Filters );

	MDSpanRO paddingDeltaRO( ctxImpl->getPaddingDelta(), paddingDeltaDims );
	if( gx_is_inner_debug ) Utils::printMDSpan( "outPadding", paddingDeltaRO );

	Dims inDeltaDims = { outDims[ 0 ], mBaseInDims[ 0 ], gx_dims_flatten_size( mBaseInDims ) / mBaseInDims[ 0 ] };
	MDSpanRW inDeltaMS( *inDelta, inDeltaDims );

	Dims dims = { mFilterDims[ 1 ], mFilterDims[ 0 ], mFilterDims[ 2 ], mFilterDims[ 3 ] };

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		Im2Rows::input2Rows( paddingDeltaRO, n, dims, &( ctxImpl->getRows4backpropagate() ) );

		//if( gx_is_inner_debug ) Utils::printMatrix( "outPadding", ctxImpl->getRows4backpropagate() );

		for( size_t i = 0; i < mRowsOfRot180Filters.size(); i++ ) {
			for( size_t j = 0; j < ctxImpl->getRows4backpropagate().size(); j++ ) {
				inDeltaMS( n, i, j ) = gx_inner_product( std::begin( mRowsOfRot180Filters[ i ] ),
						std::begin ( ctxImpl->getRows4backpropagate()[ j ] ), mRowsOfRot180Filters[ i ].size() );
			}
		}
	}
}

void ConvExLayer :: collectGradients( BaseLayerContext * ctx ) const
{
	ConvExLayerContext * ctxImpl = dynamic_cast< ConvExLayerContext * >( ctx );

	if( ctx->getGradients().size() <= 0 ) {
		ctx->getGradients().emplace_back( DataVector( gx_dims_flatten_size( mFilterDims ) ) );
	} else {
		ctx->getGradients()[ 0 ] = 0.0;
	}

	const MDSpanRO & inMS = ctx->getInMS();

	Dims deltaDims = { inMS.dim( 0 ), mBaseOutDims[ 0 ], mBaseOutDims[ 1 ], mBaseOutDims[ 2 ] };

	Dims gradientDims = { mFilterDims[ 0 ], gx_dims_flatten_size( mFilterDims ) / mFilterDims [ 0 ] };

	MDSpanRW gradientMS( ctx->getGradients()[ 0 ], gradientDims );

	for( size_t n = 0; n < inMS.dim( 0 ); n++ ) {

		Im2Rows::deltas2Rows( ctx->getDeltaMS().data(), n, deltaDims, &( ctxImpl->getRowsOfDelta() ) );

		if( gx_is_inner_debug ) Utils::printMatrix( "deltas", ctxImpl->getRowsOfDelta() );

		Im2Rows::input2Rows4Gradients( inMS, n, deltaDims, &( ctxImpl->getRows4collectGradients() ) );

		if( gx_is_inner_debug ) Utils::printMatrix( "input", ctxImpl->getRows4collectGradients() );

		for( size_t i = 0; i < ctxImpl->getRowsOfDelta().size(); i++ ) {
			for( size_t j = 0; j < ctxImpl->getRows4collectGradients().size(); j++ ) {
				gradientMS( i, j ) += gx_inner_product( std::begin( ctxImpl->getRowsOfDelta()[ i ] ),
						std::begin( ctxImpl->getRows4collectGradients()[ j ] ), ctxImpl->getRowsOfDelta()[ i ].size() );
			}
		}
	}
}

void ConvExLayer :: applyGradients( const BackwardContext & ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount )
{
	ConvLayer::applyGradients( ctx, optim, trainingCount, miniBatchCount );

	updateFiltersRows( mFilters, mFilterDims, &mRowsOfFilters, &mRowsOfRot180Filters );
}


void ConvExLayer :: updateFiltersRows( const DataVector & filters, const Dims & filterDims,
		DataMatrix * rowsOfFilters, DataMatrix * rowsOfRot180Filters )
{
	Im2Rows::filters2Rows( filters, filterDims, rowsOfFilters );
	Im2Rows::rot180Filters2Rows( filters, filterDims, rowsOfRot180Filters );

	if( gx_is_inner_debug ) Utils::printMatrix( "filters", *rowsOfFilters );
	if( gx_is_inner_debug ) Utils::printMatrix( "rot180filters", *rowsOfRot180Filters );
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
	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	MDSpanRW & outMS = ctx->getOutMS();

	outMS.dims() = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	outMS.data().resize( gx_dims_flatten_size( ctx->getOutMS().dims() ) );

	const Dims & outDims = outMS.dims();

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = pool( inMS, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType MaxPoolLayer :: pool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 1.0 * INT_MIN;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result = std::max( result, inMS( sampleIndex, filterIndex, beginX + x, beginY + y ) );
		}
	}

	return result;
}

void MaxPoolLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	MDSpanRW & outMS = ctx->getOutMS();
	const Dims & outDims = outMS.dims();

	inDelta->resize( inMS.data().size() );
	*inDelta = 0;

	MDSpanRW inDeltaMS( *inDelta, inDims );

	const MDSpanRO & outDeltaMS = ctx->getDeltaRO();

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inMS, n, f, x * mPoolSize, y * mPoolSize,
							outMS( n, f, x, y ), outDeltaMS( n, f, x, y ), &inDeltaMS );
				}
			}
		}
	}
}

void MaxPoolLayer :: unpool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex, size_t beginX, size_t beginY,
		const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			if( inMS( sampleIndex, filterIndex, beginX + x, beginY + y ) == maxValue ) {
				( *inDeltaMS )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta;
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
	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	MDSpanRW & outMS = ctx->getOutMS();

	outMS.dims() = { inDims[ 0 ], inDims[ 1 ], inDims[ 2 ] / mPoolSize, inDims[ 3 ] / mPoolSize };
	outMS.data().resize( gx_dims_flatten_size( ctx->getOutMS().dims() ) );

	const Dims & outDims = outMS.dims();

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) { 
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					outMS( n, f, x, y ) = pool( inMS, n, f, x * mPoolSize, y * mPoolSize );
				}
			}
		}
	}
}

DataType AvgPoolLayer :: pool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY ) const
{
	DataType result = 0;

	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			result += inMS( sampleIndex, filterIndex, beginX + x, beginY + y );
		}
	}

	return result / ( mPoolSize * mPoolSize );
}

void AvgPoolLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	const MDSpanRO & inMS = ctx->getInMS();
	const Dims & inDims = inMS.dims();

	MDSpanRW & outMS = ctx->getOutMS();
	const Dims & outDims = outMS.dims();

	inDelta->resize( inMS.data().size() );
	*inDelta = 0;

	MDSpanRW inDeltaMS( *inDelta, inDims );

	const MDSpanRO & outDeltaMS = ctx->getDeltaRO();

	for( size_t n = 0; n < outDims[ 0 ]; n++ ) {
		for( size_t f = 0; f < outDims[ 1 ]; f++ ) {
			for( size_t x = 0; x < outDims[ 2 ]; x++ ) {
				for( size_t y = 0; y < outDims[ 3 ]; y++ ) {
					unpool( inMS, n, f, x * mPoolSize, y * mPoolSize,
							outMS( n, f, x, y ), outDeltaMS( n, f, x, y ), &inDeltaMS );
				}
			}
		}
	}
}

void AvgPoolLayer :: unpool( const MDSpanRO & inMS, size_t sampleIndex, size_t filterIndex,
		size_t beginX, size_t beginY, const DataType maxValue, DataType outDelta, MDSpanRW * inDeltaMS ) const
{
	for( size_t x = 0; x < mPoolSize; x++ ) {
		for( size_t y = 0; y < mPoolSize; y++ ) {
			( *inDeltaMS )( sampleIndex, filterIndex, beginX + x, beginY + y ) = outDelta / ( mPoolSize * mPoolSize );
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

	const DataVector & input = ctx->getInMS().data();
	DataVector & output = ctx->getOutMS().data();

	ctx->getOutMS().dims() = ctx->getInMS().dims();
	ctx->getOutMS().data().resize( input.size() );

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
		Utils::printMDSpan( "dropout.input", ctx->getInMS() );
		Utils::printMDSpan( "dropout.output", ctx->getOutRO() );
	}
}

void DropoutLayer :: backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const
{
	DropoutLayerContext * ctxImpl = dynamic_cast< DropoutLayerContext * >( ctx );

	assert( NULL != ctxImpl );

	const DataVector & delta = ctx->getDeltaRO().data();

	for( size_t i = 0; i < delta.size(); i++ )
			( *inDelta )[ i ] = ctxImpl->getMask()[ i ] ? 0 : delta[ i ];

	if( gx_is_inner_debug ) {
		Utils::printMDSpan( "dropout.outDelta", ctx->getDeltaRO() );
		Utils::printVector( "dropout.inDelta", *inDelta );
	}
}


}; // namespace gxnet;

