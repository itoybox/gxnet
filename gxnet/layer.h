#pragma once

#include "common.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace gxnet {

class BaseLayer;
class BaseLayerCtx;
class BackwardCtx;

typedef std::vector< BaseLayer * > BaseLayerPtrVector;

class ActFunc;
class Optim;

class BaseLayer {
public:
	enum { eFullConn = 1, };

public:
	BaseLayer( int type );

	virtual ~BaseLayer();

	virtual void collectGradients( BaseLayerCtx * ctx ) const;

	virtual void applyGradients( BackwardCtx * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

public:

	virtual void print( bool isDetail = false ) const;

	void forward( BaseLayerCtx * ctx ) const;

	void backward( BaseLayerCtx * ctx, DataVector * inDelta ) const;

	BaseLayerCtx * createCtx( const DataVector * input ) const;

protected:

	virtual void printWeights( bool isDetail ) const = 0;

	virtual BaseLayerCtx * newCtx( const DataVector * input ) const = 0;

	virtual void calcOutput( BaseLayerCtx * ctx ) const = 0;

	virtual void backpropagate( BaseLayerCtx * ctx, DataVector * inDelta ) const = 0;

public:
	int getType() const;

	const Dims & getInputDims() const;

	const size_t getInputSize() const;

	const Dims & getOutputDims() const;

	const size_t getOutputSize() const;

	void setActFunc( ActFunc * actFunc );

	const ActFunc * getActFunc() const;

	void setTraining( bool isTraining );

protected:
	Dims mInputDims, mOutputDims;
	int mType;
	bool mIsTraining;

	ActFunc * mActFunc;
};

class FullConnLayer : public BaseLayer {
public:
	FullConnLayer( const Dims & inputDims, size_t neuronCount );

	~FullConnLayer();

	// for debug
	void setWeights( const DataMatrix & weights, const DataVector & biases );

	const DataMatrix & getWeights() const;

	const DataVector & getBiases() const;

	virtual void collectGradients( BaseLayerCtx * ctx ) const;

	virtual void applyGradients( BackwardCtx * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerCtx * newCtx( const DataVector * input ) const;

	virtual void calcOutput( BaseLayerCtx * ctx ) const;

	virtual void backpropagate( BaseLayerCtx * ctx, DataVector * inDelta ) const;

private:
	DataMatrix mWeights;
	DataVector mBiases;
};

}; // namespace gxnet;

