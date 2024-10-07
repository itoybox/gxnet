from __future__ import print_function

import struct
import numpy as np
import random
import ctypes
import sys
import random

from PIL import Image

def print_mnist( data ):
	tmp = data.reshape( -1 )
	for i in range( 28 ):
		for j in range( 28 ):
			if tmp[ i * 28 + j ] == 0 : print( "0", end="" )
			else: print( "1", end="" )
			#print( tmp[ i * 28 + j ], " ", end="" )
		print( "" )

def read_mnist_images( path ):

	fp = open( path, "rb" )
	buff = fp.read()
	fp.close()

	offset = 0

	fmt_header = '>iiii'
	magic_number, num_images, num_rows, num_cols = struct.unpack_from( fmt_header, buff, offset )

	print( "load {}, magic {}, count {}".format( path, magic_number, num_images ) )

	offset += struct.calcsize( fmt_header )
	fmt_image = '>' + str( num_rows * num_cols ) + 'B'

	images = np.empty( ( num_images, num_rows, num_cols ) )

	for i in range( num_images ):
		im = struct.unpack_from( fmt_image, buff, offset )
		images[ i ] = np.array( im ).reshape( ( num_rows, num_cols ) )
		offset += struct.calcsize( fmt_image )
	return images

def write_mnist_images( path, images ):

	fp = open( path, "wb" )

	buff = struct.pack( '>IIII', 2051, len( images ), 28, 28 )
	fp.write( buff )

	fmt_image = '>784B'

	for img in images:
		new_array = np.array( img )
		new_buff = new_array.astype( np.uint8 ).ravel()

		buff = struct.pack( fmt_image, *new_buff )
		fp.write( buff )

	fp.close()

def read_mnist_labels( path ):
	fp = open( path, "rb" )
	buff = fp.read()
	fp.close()

	offset = 0

	fmt_header = '>ii'
	magic_number, label_num = struct.unpack_from(fmt_header, buff, offset)

	print( "load {}, magic {}, count {}".format( path, magic_number, label_num ) )

	offset += struct.calcsize(fmt_header)
	labels = []

	fmt_label = '>B'

	for i in range( label_num ):
		labels.append( struct.unpack_from( fmt_label, buff, offset )[ 0 ] )
		offset += struct.calcsize( fmt_label )
	return labels

def write_mnist_labels( path, labels ):

	fp = open( path, "wb" )

	buff = struct.pack( '>II', 2049, len( labels ) )
	fp.write( buff )

	fmt_image = '>748B'

	for label in labels:
		buff = struct.pack( '>B', label )
		fp.write( buff )

	fp.close()

def split( image_path, label_path ):

	images = read_mnist_images( image_path )
	labels = read_mnist_labels( label_path )

	idx = [ i for i in range( len( labels ) ) ]

	random.shuffle( idx )

	train_stats = {}
	test_stats = {}
	unused_stats = {}

	train_images = []
	train_labels = []

	test_images = []
	test_labels = []

	for i in idx:
		label = labels[ i ]

		is_picked = False

		if label in test_stats:
			if test_stats[ label ] < 300:
				test_stats[ label ] +=1
				is_picked = True
		else:
			test_stats[ label ] = 1
			is_picked = True

		if is_picked:
			test_images.append( images[ i ] )
			test_labels.append( labels[ i ] )
			continue

		is_picked = False

		if label in train_stats:
			if train_stats[ label ] < 6000:
				train_stats[ label ] += 1
				is_picked = True
		else:
			train_stats[ label ] = 1
			is_picked = True

		if is_picked:
			train_images.append( images[ i ] )
			train_labels.append( labels[ i ] )
		else:
			if label in unused_stats:
				unused_stats[ label ] += 1
			else:
				unused_stats[ label ] = 1

	print( "train_stats", train_stats )
	print( "test_stats", test_stats )
	print( "unused_stats", unused_stats )

	write_mnist_images( image_path + ".train", train_images )
	write_mnist_labels( label_path + ".train", train_labels )

	write_mnist_images( image_path + ".test", test_images )
	write_mnist_labels( label_path + ".test", test_labels )

if __name__ == '__main__':

	if( len( sys.argv ) < 3 ):
		print( "Usage: %s <mnist images file> <mnist labels file>\n" % ( sys.argv[ 0 ] ) )
		sys.exit( -1 )

	split( sys.argv[ 1 ], sys.argv[ 2] )

