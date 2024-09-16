
#include "activation.h"

namespace gxnet {

ActFunc * ActFunc :: sigmoid()
{
	return new ActFunc( eSigmoid );
}

ActFunc * ActFunc :: tanh()
{
	return new ActFunc( eTanh );
}

ActFunc * ActFunc :: leakyReLU()
{
	return new ActFunc( eLeakyReLU );
}

ActFunc * ActFunc :: softmax()
{
	return new ActFunc( eSoftmax );
}

ActFunc :: ActFunc( int type )
{
	mType = type;
}

ActFunc :: ~ActFunc()
{
}

int ActFunc :: getType() const
{
	return mType;
}

void ActFunc :: activate( const DataVector & input, DataVector * output ) const
{
	if( eSigmoid == mType ) {
		*output = 1.0f / ( 1.0f + std::exp( - input ) );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < input.size(); i++ ) {
			if( input[ i ] < 0 ) {
				( *output )[ i ] = 0.01 * input[ i ];
			} else if( input[ i ] > 1 ) {
				( *output )[ i ] = 1 + 0.01 * ( input[ i ] - 1 );
			}
		}
	}

	if( eTanh == mType ) {
		*output = std::tanh( input );
	}

	if( eSoftmax == mType ) {
		*output = std::exp( input - input.max() );
		*output /= output->sum();
	}
}

void ActFunc :: derivate( const DataVector & output, DataVector * outDelta ) const
{
	if( eSigmoid == mType ) {
		*outDelta = output * ( 1 - output ) * ( *outDelta );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			( *outDelta )[ i ] = ( *outDelta )[ i ] * ( output[ i ] < 0 || output[ i ] > 1 ? 0.01 : 1 );
		}
	}

	if( eTanh == mType ) {
		*outDelta = ( *outDelta ) * ( 1 - output * output );
	}

	if( eSoftmax == mType ) {
		DataVector dOutput = *outDelta;
		for( size_t j = 0; j < output.size(); j++ ) {
			DataVector dSoftmax( output.size() );
			for( size_t k = 0; k < output.size(); k++ ) {
				dSoftmax[ k ] = ( k == j ) ? output[ j ] * ( 1.0 - output[ j ] ) : -output[ k ] * output[ j ];
			}

			//( *outDelta )[ j ] = ( dOutput * dSoftmax ).sum();
			( *outDelta )[ j ] = gx_inner_product( dOutput, dSoftmax );
		}
	}
}


}; // namespace gxnet;

