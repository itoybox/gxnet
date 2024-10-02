#pragma once

#include "common.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace gxnet {

class BaseLayer;
class BaseLayerContext;
class BackwardContext;

typedef std::vector< BaseLayer * > BaseLayerPtrVector;

class ActFunc;
class Optim;

class BaseLayer {
public:
	enum { eFullConn = 1, };

public:
	BaseLayer( int type );

	virtual ~BaseLayer();

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( BackwardContext * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

public:

	virtual void print( bool isDetail = false ) const;

	void forward( BaseLayerContext * ctx ) const;

	void backward( BaseLayerContext * ctx, DataVector * inDelta ) const;

	BaseLayerContext * createCtx() const;

protected:

	virtual void printWeights( bool isDetail ) const = 0;

	virtual BaseLayerContext * newCtx() const = 0;

	virtual void calcOutput( BaseLayerContext * ctx ) const = 0;

	virtual void backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const = 0;

public:
	int getType() const;

	void setActFunc( ActFunc * actFunc );

	const ActFunc * getActFunc() const;

	void setTraining( bool isTraining );

protected:
	int mType;
	bool mIsTraining;

	ActFunc * mActFunc;
};

class FullConnLayer : public BaseLayer {
public:
	FullConnLayer( size_t neuronCount, size_t inSize );

	~FullConnLayer();

	// for debug
	void setWeights( const DataMatrix & weights, const DataVector & biases );

	const DataMatrix & getWeights() const;

	const DataVector & getBiases() const;

	virtual void collectGradients( BaseLayerContext * ctx ) const;

	virtual void applyGradients( BackwardContext * ctx, Optim * optim,
			size_t trainingCount, size_t miniBatchCount );

protected:

	virtual void printWeights( bool isDetail ) const;

	virtual BaseLayerContext * newCtx() const;

	virtual void calcOutput( BaseLayerContext * ctx ) const;

	virtual void backpropagate( BaseLayerContext * ctx, DataVector * inDelta ) const;

private:
	DataMatrix mWeights;
	DataVector mBiases;
};

}; // namespace gxnet;

