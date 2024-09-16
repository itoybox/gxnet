
#include "network.h"
#include "utils.h"

#include <random>
#include <numeric>
#include <algorithm>

#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>

namespace gxnet {

NetworkCtx :: NetworkCtx()
{
}

NetworkCtx :: ~NetworkCtx()
{
	for( auto & item : mLayerCtx ) delete item;
	for( auto & item : mBatchCtx ) delete item;
}

void NetworkCtx :: setTrainingData( const DataMatrix * input, const DataMatrix * target )
{
	mInput = input;
	mTarget = target;
}

const DataMatrix & NetworkCtx :: getInput()
{
	return *mInput;
}

const DataMatrix & NetworkCtx :: getTarget()
{
	return *mTarget;
}

void NetworkCtx :: setChunk( const IntVector * idxOfData, size_t begin, size_t end )
{
	mIdxOfData = idxOfData;
	mChunkBegin = begin;
	mChunkEnd = end;
}

const IntVector & NetworkCtx :: getIdxOfData()
{
	return *mIdxOfData;
}

size_t NetworkCtx :: getChunkBegin()
{
	return mChunkBegin;
}

size_t NetworkCtx :: getChunkEnd()
{
	return mChunkEnd;
}

BaseLayerCtx * NetworkCtx :: getLayerCtx( size_t index )
{
	return mLayerCtx[ index ];
}

BaseLayerCtxPtrVector & NetworkCtx :: getLayerCtx()
{
	return mLayerCtx;
}

BackwardCtx * NetworkCtx :: getBatchCtx( size_t index )
{
	return mBatchCtx[ index ];
}

BackwardCtxPtrVector & NetworkCtx :: getBatchCtx()
{
	return mBatchCtx;
}

void NetworkCtx :: clearBatch()
{
	for( auto & ctx : mBatchCtx ) {
		for( auto & vec : ctx->getGradients() ) vec = 0.0;
		ctx->getDelta() = 0.0;
	}
}

void NetworkCtx :: addToBatch()
{
	for( size_t i = 0; i < mBatchCtx.size(); i++ ) {
		mBatchCtx[ i ]->getDelta() += mLayerCtx[ i ]->getBackwardCtx().getDelta();

		for( size_t j = 0; j < mBatchCtx[ i ]->getGradients().size(); j++ ) {
			mBatchCtx[ i ]->getGradients()[ j ] += mLayerCtx[ i ]->getBackwardCtx().getGradients()[ j ];
		}
	}
}

////////////////////////////////////////////////////////////

Network :: Network( int lossFuncType )
{
	mOnEpochEnd = NULL;
	mLossFuncType = lossFuncType;
	mIsDebug = false;
	mIsShuffle = true;
	mIsTraining = false;
}

Network :: ~Network()
{
	for( auto & item : mLayers ) delete item;
}

void Network :: print( bool isDetail ) const
{
	printf( "\n{{{ isDetail %s\n", isDetail ? "true" : "false" );
	printf( "Network: LayerCount = %zu; LossFuncType = %d;\n", mLayers.size(), mLossFuncType );
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		printf( "\nLayer#%ld: ", i );
		layer->print( isDetail );
	}
	printf( "}}}\n\n" );
}

void Network :: setOnEpochEnd( OnEpochEnd_t onEpochEnd )
{
	mOnEpochEnd = onEpochEnd;
}

void Network :: setDebug( bool isDebug )
{
	mIsDebug = isDebug;

	for( auto & layer : mLayers ) layer->setDebug( isDebug );
}

void Network :: setTraining( bool isTraining )
{
	mIsTraining = isTraining;

	for( auto & layer : mLayers ) layer->setTraining( isTraining );
}

void Network :: setShuffle( bool isShuffle )
{
	mIsShuffle = isShuffle;
}

void Network :: setLossFuncType( int lossFuncType )
{
	mLossFuncType = lossFuncType;
}

int Network :: getLossFuncType() const
{
	return mLossFuncType;
}

BaseLayerPtrVector & Network :: getLayers()
{
	return mLayers;
}

const BaseLayerPtrVector & Network :: getLayers() const
{
	return mLayers;
}

void Network :: addLayer( BaseLayer * layer )
{
	layer->setDebug( mIsDebug );
	layer->setTraining( mIsTraining );

	mLayers.emplace_back( layer );
}

bool Network :: forward( const DataVector & input, DataVector * output ) const
{
	NetworkCtx ctx;
	initCtx( &ctx );

	ctx.getLayerCtx( 0 )->getForwardCtx().setInput( &input );

	bool ret = forward( &ctx );

	*output = ctx.getLayerCtx().back()->getForwardCtx().getOutput();

	return ret;
}

bool Network :: forward( NetworkCtx * ctx ) const
{
	if( ctx->getLayerCtx(0)->getForwardCtx().getInput().size() != mLayers[ 0 ]->getInputSize() ) {
		printf( "%s input.size %zu, layer[0].inputSize %zu",
				__func__, ctx->getLayerCtx(0)->getForwardCtx().getInput().size(),
				mLayers[ 0 ]->getInputSize() );
		return false;
	}

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		BaseLayerCtx * layerCtx = ctx->getLayerCtx( i );

		layer->forward( layerCtx );
	}

	return true;
}

bool Network :: apply( NetworkCtx * ctx, Optim * optim,
		size_t trainingCount, size_t miniBatchCount )
{
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];
		layer->applyGradients( ctx->getBatchCtx( i ), optim, trainingCount, miniBatchCount );
	}

	return true;
}

void Network :: collect( NetworkCtx * ctx ) const
{
	for( size_t i = 0; i < mLayers.size(); i++ ) {
		BaseLayer * layer = mLayers[ i ];

		layer->collectGradients( ctx->getLayerCtx( i ) );
	}

	if( mIsDebug ) {
		Utils::printVector( "input", ctx->getLayerCtx( 0 )->getForwardCtx().getInput() );
		Utils::printCtx( "collect", ctx->getLayerCtx() );
	}
}

bool Network :: backward( NetworkCtx * ctx, const DataVector & target ) const
{
	BaseLayerCtx * layerCtx = ctx->getLayerCtx().back();

	const DataVector & lastOutput = layerCtx->getForwardCtx().getOutput();

	if( eMeanSquaredError == mLossFuncType ) {
		layerCtx->getBackwardCtx().getDelta() = 2.0 * ( lastOutput - target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		layerCtx->getBackwardCtx().getDelta() = lastOutput - target;
	}

	for( ssize_t i = mLayers.size() - 1; i >= 0; i-- ) {
		DataVector * inDelta = ( i > 0 ) ? &( ctx->getLayerCtx()[ i - 1 ]->getBackwardCtx().getDelta() ) : NULL;

		BaseLayer * layer = mLayers[ i  ];

		layer->backward( ctx->getLayerCtx()[ i ], inDelta );
	}

	return true;
}

void Network :: initCtx( NetworkCtx * ctx ) const
{
	BaseLayerCtx * layerCtx = NULL;

	for( auto & layer : mLayers ) {
		layerCtx = layer->createCtx( NULL == layerCtx ? NULL : ( & layerCtx->getForwardCtx().getOutput() ) );
		ctx->getLayerCtx().push_back( layerCtx );
	}

	for( size_t i = 0; i < mLayers.size(); i++ ) {
		ctx->getBatchCtx().emplace_back(
				new BackwardCtx( ctx->getLayerCtx( i )->getBackwardCtx() ) );
	}
}

DataType Network :: calcLoss( const DataVector & target, const DataVector & output )
{
	DataType ret = 0;

	if( eMeanSquaredError == mLossFuncType ) {
		ret = Utils::calcSSE( output, target );
	}

	if( eCrossEntropy == mLossFuncType ) {
		for( size_t x = 0; x < target.size(); x++ ) {
			DataType y = target[ x ], a = output[ x ];
			DataType tmp = y * std::log( a ); // + ( 1 - y ) * std::log( 1 - a );
			ret -= tmp;
		}
	}

	return ret;
}

bool Network :: trainMiniBatch( NetworkCtx * ctx, DataType * totalLoss )
{
	for( size_t i = ctx->getChunkBegin(); i < ctx->getChunkEnd(); i++ ) {

		const DataVector & currInput = ctx->getInput()[ ctx->getIdxOfData()[ i ] ];
		const DataVector & currTarget = ctx->getTarget()[ ctx->getIdxOfData()[ i ] ];

		ctx->getLayerCtx( 0 )->getForwardCtx().setInput( &currInput );

		forward( ctx );

		backward( ctx, currTarget );

		collect( ctx );

		ctx->addToBatch();

		DataType loss = calcLoss( currTarget, ctx->getLayerCtx().back()->getForwardCtx().getOutput() );

		*totalLoss += loss;

		if( mIsDebug )  printf( "DEBUG: input #%ld loss %.8f totalLoss %.8f\n", i, loss, *totalLoss );
	}

	return true;
}

bool Network :: trainInternal( const DataMatrix & input, const DataMatrix & target,
		const CmdArgs_t & args, DataVector * losses )
{
	if( input.size() != target.size() ) return false;

	setTraining( true );

	time_t beginTime = time( NULL );

	printf( "%s\tstart train, input { %zu }, target { %zu }\n",
			ctime( &beginTime ), input.size(), target.size() );

	int logInterval = args.mEpochCount / 10;
	int progressInterval = ( input.size() / args.mMiniBatchCount ) / 10;

	std::random_device rd;
	std::mt19937 gen( rd() );

	assert( mLayers[ 0 ]->getInputSize() == input[ 0 ].size() );

	NetworkCtx ctx;
	initCtx( &ctx );
	ctx.setTrainingData( &input, &target );

	std::unique_ptr< Optim > optim( Optim::SGD( args.mLearningRate, args.mLambda ) );
	optim->setDebug( mIsDebug );

	if( NULL != losses ) losses->resize( args.mEpochCount, 0 );

	for( int n = 0; n < args.mEpochCount; n++ ) {

		IntVector idxOfData( input.size() );
		std::iota( idxOfData.begin(), idxOfData.end(), 0 );
		if( mIsShuffle ) std::shuffle( idxOfData.begin(), idxOfData.end(), gen );

		DataType totalLoss = 0;

		int miniBatchCount = std::max( args.mMiniBatchCount, 1 );

		for( size_t begin = 0; begin < idxOfData.size(); ) {
			size_t end = std::min( idxOfData.size(), begin + miniBatchCount );

			ctx.clearBatch();

			ctx.setChunk( &idxOfData, begin, end );

			trainMiniBatch( &ctx, &totalLoss );

			if( mIsDebug ) {
				Utils::printCtx( "batch", ctx.getBatchCtx() );
			}

			apply( &ctx, optim.get(), input.size(), end - begin );

			if( progressInterval > 0 && 0 == ( begin % ( progressInterval * miniBatchCount ) ) ) {
				printf( "\r%zu / %zu, loss %.8f", end, idxOfData.size(), totalLoss / end );
				fflush( stdout );
			}

			begin += miniBatchCount;
			end = begin + miniBatchCount;
		}

		if( NULL != losses ) ( *losses )[ n ] = totalLoss / input.size();

		if( logInterval <= 1 || ( logInterval > 1 && 0 == n % logInterval ) || n == ( args.mEpochCount - 1 ) ) {
			time_t currTime = time( NULL );
			printf( "\33[2K\r%s\tinterval %ld [>] epoch %d, lr %f, loss %.8f\n",
				ctime( &currTime ), currTime - beginTime, n, args.mLearningRate, totalLoss / input.size() );
			beginTime = time( NULL );
		}

		if( mIsDebug ) print();

		if( mOnEpochEnd ) mOnEpochEnd( *this, n, totalLoss / input.size() );
	}

	return true;
}

bool Network :: train( const DataMatrix & input, const DataMatrix & target,
		const CmdArgs_t & args, DataVector * losses )
{
	std::chrono::steady_clock::time_point beginTime = std::chrono::steady_clock::now();	

	bool ret = trainInternal( input, target, args, losses );

	std::chrono::steady_clock::time_point endTime = std::chrono::steady_clock::now();	

	auto timeSpan = std::chrono::duration_cast<std::chrono::milliseconds>( endTime - beginTime );

	printf( "Elapsed time: %.3f\n", timeSpan.count() / 1000.0 );

	return ret;
}


}; // namespace gxnet;

