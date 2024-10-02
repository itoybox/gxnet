
#include "optim.h"

//#include <execution>

namespace gxnet {

Optim :: Optim( int type )
{
	mType = type;
}

Optim :: ~Optim()
{
}

int Optim :: getType() const
{
	return mType;
}

////////////////////////////////////////////////////////////

class SGD : public Optim {
public:
	SGD( DataType lr, DataType lambda );

	virtual ~SGD();

	virtual void update( DataVector * weights, const DataVector & gradients,
			size_t trainingCount, size_t miniBatchCount );

	virtual void updateBiases( DataVector * bias, const DataVector & delta,
			size_t miniBatchCount );
private:
	DataType mLR, mLambda;
};

SGD :: SGD( DataType lr, DataType lambda )
	: Optim( eSGD )
{
	mLR = lr;
	mLambda = lambda;
}

SGD :: ~SGD()
{
}

void SGD :: update( DataVector * weights, const DataVector & gradients,
		size_t trainingCount, size_t miniBatchCount )
{
	DataVector tmpGrad = gradients;

	if( gx_is_inner_debug ) {
		gx_vs_product( tmpGrad, mLR, &tmpGrad );
	} else {
		gx_vs_product( *weights, ( 1.0 - mLR * mLambda / trainingCount ), weights );
		gx_vs_product( tmpGrad, mLR / miniBatchCount, &tmpGrad );
	}

	*weights -= tmpGrad;

	//std::transform( std::execution::par, std::begin( *weights ), std::end( *weights ),
			///std::begin( tmpGrad ), std::begin( *weights ), std::minus{} );
}

void SGD :: updateBiases( DataVector * bias, const DataVector & delta,
		size_t miniBatchCount )
{
	( *bias ) = ( *bias ) - delta * mLR / miniBatchCount;
}

////////////////////////////////////////////////////////////

Optim * Optim :: SGD( DataType lr, DataType lambda )
{
	return new class SGD( lr, lambda );
}


}; // namespace gxnet;

