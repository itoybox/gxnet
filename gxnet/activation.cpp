
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

void ActFunc :: activate( const MDSpanRO & inMS, MDSpanRW * outMS ) const
{
	const DataVector & input = inMS.data();
	DataVector & output = outMS->data();

	if( eSigmoid == mType ) {
		output = 1.0f / ( 1.0f + std::exp( - input ) );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < input.size(); i++ ) {
			if( input[ i ] < 0 ) {
				output[ i ] = 0.01 * input[ i ];
			} else if( input[ i ] > 1 ) {
				output[ i ] = 1 + 0.01 * ( input[ i ] - 1 );
			}
		}
	}

	if( eTanh == mType ) {
		output = std::tanh( input );
	}

	if( eSoftmax == mType ) {
		size_t total = gx_dims_flatten_size( inMS.dims() );
		size_t inSize = total / inMS.dims()[ 0 ];

		const DataType * inPtr = std::begin( input );
		DataType * outPtr = std::begin( output );

		DataVector tmpIn( inSize ), tmpOut( inSize );
		for( size_t index = 0; index < total; index += inSize, inPtr += inSize, outPtr += inSize ) {

			std::copy( inPtr, inPtr + inSize, std::begin( tmpIn ) );

			tmpOut = std::exp( tmpIn - tmpIn.max() );
			tmpOut /= tmpOut.sum();

			std::copy( std::begin( tmpOut ), std::end( tmpOut ), outPtr );
		}
	}
}

void ActFunc :: derivate( const MDSpanRO & outMS, MDSpanRW * outDeltaMS ) const
{
	const DataVector & output = outMS.data();
	DataVector & outDelta = outDeltaMS->data();

	if( eSigmoid == mType ) {
		outDelta = output * ( 1 - output ) * ( outDelta );
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < output.size(); i++ ) {
			outDelta[ i ] = outDelta[ i ] * ( output[ i ] < 0 || output[ i ] > 1 ? 0.01 : 1 );
		}
	}

	if( eTanh == mType ) {
		outDelta = outDelta * ( 1 - output * output );
	}

	if( eSoftmax == mType ) {
		size_t total = gx_dims_flatten_size( outMS.dims() );
		size_t outSize = total / outMS.dims()[ 0 ];

		DataVector dOutput( outSize ), dSoftmax( outSize );

		DataType * outDeltaPtr = std::begin( outDelta );

		for( size_t index = 0; index < total; index += outSize, outDeltaPtr += outSize ) {
			std::copy( outDeltaPtr, outDeltaPtr + outSize, std::begin( dOutput ) );

			for( size_t j = 0; j < outSize; j++ ) {
				for( size_t k = 0; k < outSize; k++ ) {
					dSoftmax[ k ] = ( k == j ) 
							?
							output[ index + j ] * ( 1.0 - output[ index + j ] )
							:
							-output[ index + k ] * output[ index + j ];
				}

				outDelta[ index + j ] = gx_inner_product( std::begin( dOutput ), std::begin( dSoftmax ), dOutput.size() );
			}
		}
	}
}


}; // namespace gxnet;

