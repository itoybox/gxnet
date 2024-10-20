
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

void ActFunc :: activate( const MDVector & inMD, MDVector * outMD ) const
{
	const DataType * input = std::begin( inMD.first );
	DataType * output = std::begin( outMD->first );

	size_t total = gx_dims_flatten_size( inMD.second );

	if( eSigmoid == mType ) {
		//output = 1.0f / ( 1.0f + std::exp( - input ) );
		std::transform( input, input + total, output, output,
				[]( const DataType & a, const DataType & b ) {
					return 1.0f / ( 1.0f + std::exp( - a ) );
				}
		);
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < total; i++ ) {
			if( input[ i ] < 0 ) {
				output[ i ] = 0.01 * input[ i ];
			} else if( input[ i ] > 1 ) {
				output[ i ] = 1 + 0.01 * ( input[ i ] - 1 );
			}
		}
	}

	if( eTanh == mType ) {
		//output = std::tanh( input );
		std::transform( input, input + total, output, output,
				[]( const DataType & a, const DataType & b ) {
					return std::tanh( a );
				}
		);
	}

	if( eSoftmax == mType ) {
		size_t inSize = total / inMD.second[ 0 ];

		const DataType * inPtr = input;
		DataType * outPtr = output;

		DataVector tmpIn( inSize ), tmpOut( inSize );
		for( size_t index = 0; index < total; index += inSize, inPtr += inSize, outPtr += inSize ) {

			std::copy( inPtr, inPtr + inSize, std::begin( tmpIn ) );

			tmpOut = std::exp( tmpIn - tmpIn.max() );
			tmpOut /= tmpOut.sum();

			std::copy( std::begin( tmpOut ), std::end( tmpOut ), outPtr );
		}
	}
}

void ActFunc :: derivate( const MDVector & outMD, MDVector * outDeltaMD ) const
{
	const DataType * output = std::begin( outMD.first );
	DataType * outDelta = std::begin( outDeltaMD->first );

	size_t total = gx_dims_flatten_size( outMD.second );

	if( eSigmoid == mType ) {
		//outDelta = output * ( 1 - output ) * ( outDelta );
		std::transform( output, output + total, outDelta, outDelta,
				[]( const DataType & a, const DataType & b ) {
					return a * ( 1 - a ) * b;
				}
		);
	}

	if( eLeakyReLU == mType ) {
		for( size_t i = 0; i < total; i++ ) {
			outDelta[ i ] = outDelta[ i ] * ( output[ i ] < 0 || output[ i ] > 1 ? 0.01 : 1 );
		}
	}

	if( eTanh == mType ) {
		//outDelta = outDelta * ( 1 - output * output );
		std::transform( output, output + total, outDelta, outDelta,
				[]( const DataType & a, const DataType & b ) {
					return b * ( 1 - a * a );
				}
		);
	}

	if( eSoftmax == mType ) {
		size_t outSize = total / outMD.second[ 0 ];

		DataVector dOutput( outSize ), dSoftmax( outSize );

		DataType * outDeltaPtr = outDelta;

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

				outDelta[ index + j ] = gx_inner_product( std::begin( dOutput ),
						std::begin( dSoftmax ), dOutput.size() );
			}
		}
	}
}


}; // namespace gxnet;

