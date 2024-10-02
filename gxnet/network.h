#pragma once

#include "common.h"
#include "layer.h"
#include "context.h"
#include "optim.h"
#include "utils.h"

namespace gxnet {

class Network;

typedef void ( * OnEpochEnd_t )( Network & network, int epoch, DataType loss );

class NetworkCtx {
public:
	NetworkCtx();
	~NetworkCtx();

	void setTrainingData( const DataMatrix * input, const DataMatrix * target );

	const DataMatrix & getInput();

	const DataMatrix & getTarget();

	void setChunk( const IntVector * idxOfData, size_t begin, size_t end );

	const IntVector & getIdxOfData();

	size_t getChunkBegin();

	size_t getChunkEnd();

	BaseLayerCtx * getLayerCtx( size_t index );

	BaseLayerCtxPtrVector & getLayerCtx();

	BackwardCtx * getBatchCtx( size_t index );

	BackwardCtxPtrVector & getBatchCtx();

	void clearBatch();

	void addToBatch();

private:
	BaseLayerCtxPtrVector mLayerCtx;
	BackwardCtxPtrVector mBatchCtx;

	const DataMatrix * mInput, * mTarget;

	const IntVector * mIdxOfData;
	size_t mChunkBegin, mChunkEnd;
};

class Network {
public:
	enum { eMeanSquaredError = 1, eCrossEntropy = 2 };

	Network( int lossFuncType = eMeanSquaredError );
	~Network();

	void setOnEpochEnd( OnEpochEnd_t onEpochEnd );

	void setTraining( bool isTraining );

	void setLossFuncType( int lossFuncType );

	int getLossFuncType() const;

	void addLayer( BaseLayer * layer );

	BaseLayerPtrVector & getLayers();

	const BaseLayerPtrVector & getLayers() const;

	void initCtx( NetworkCtx * ctx ) const;

	bool forward( const DataVector & input, DataVector * output ) const;

	bool train( const DataMatrix & input, const DataMatrix & target,
			const CmdArgs_t & args, DataVector * losses = nullptr );

	void print( bool isDetail = false ) const;

	bool trainMiniBatch( NetworkCtx * ctx, DataType * totalLoss );

private:

	bool forward( NetworkCtx * ctx ) const;

	bool backward( NetworkCtx * ctx, const DataVector & target ) const;

	void collect( NetworkCtx * ctx ) const;

	bool apply( NetworkCtx * ctx, Optim * optim, size_t trainingCount, size_t miniBatchCount );

	DataType calcLoss( const DataVector & target, const DataVector & output );

	bool trainInternal( const DataMatrix & input, const DataMatrix & target,
			const CmdArgs_t & args, DataVector * losses = nullptr );

private:
	OnEpochEnd_t mOnEpochEnd;
	int mLossFuncType;
	BaseLayerPtrVector mLayers;
	bool mIsTraining;
};


}; // namespace gxnet;

