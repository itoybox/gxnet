from __future__ import print_function

import csv
import sys
import struct
import numpy as np

def csv2mnist( path ):

	images_path = path + ".idx3-ubyte"
	labels_path = path + ".idx1-ubyte"

	fp = open( images_path, "wb" )
	fp2 = open( labels_path, "wb" )

	buff = struct.pack( '>IIII', 2051, 0, 0, 0 )
	fp.write( buff )

	buff = struct.pack( '>II', 2049, 0 )
	fp2.write( buff )

	num_images = 0

	rows, cols = 28, 28
	fmt_image = '>' + str( rows * cols ) + 'B'

	with open( path ) as f:
		for row in csv.reader( f ):
			new_array = np.array( row[ 1 : ] )
			new_buff = new_array.astype( np.uint8 ).ravel()

			buff = struct.pack( fmt_image, *new_buff )
			fp.write( buff )

			buff = struct.pack( '>B', int( row[ 0 ] ) )
			fp2.write( buff )

			num_images += 1

	fp.seek( 0, 0 )
	buff = struct.pack( '>IIII', 2051, num_images, rows, cols )
	fp.write( buff )
	fp.close()

	fp2.seek( 0, 0 )
	buff = struct.pack( '>II', 2049, num_images )
	fp2.write( buff )
	fp2.close()

	print( "convert %s to %s,%s" % ( path, images_path, labels_path ) )

if __name__ == '__main__':

	if( len( sys.argv ) < 2 ):
		print( "Usage: %s <csv file>\n" % ( sys.argv[ 0 ] ) )
		sys.exit( -1 )

	csv2mnist( sys.argv[ 1 ] )

