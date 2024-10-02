
#include "optim.h"

#include <execution>

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
		gx_vs_product( std::begin( tmpGrad ), mLR, std::begin( tmpGrad ), tmpGrad.size() );
	} else {
		gx_vs_product( std::begin( *weights ), ( 1.0 - mLR * mLambda / trainingCount ),
				std::begin( *weights ), weights->size() );
		gx_vs_product( std::begin( tmpGrad ), mLR / miniBatchCount,
				std::begin( tmpGrad ), tmpGrad.size() );
	}

	//*weights -= tmpGrad;

	std::transform( std::begin( *weights ), std::end( *weights ),
			std::begin( tmpGrad ), std::begin( *weights ), std::minus{} );
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

