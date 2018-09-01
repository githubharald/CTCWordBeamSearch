#!/bin/bash


# check if parallel decoding is enabled: specify PARALLEL NUMTHREADS, e.g. PARALLEL 8
if [ "$1" == "PARALLEL" ]; then

	# default to 4 threads if not specified
	if [ -z "$2" ]; then
		NUMTHREADS="4"
	else
		NUMTHREADS=$2
	fi

	echo "Parallel decoding with $NUMTHREADS threads"
	PARALLEL="-DWBS_PARALLEL -DWBS_THREADS=$NUMTHREADS"
else
	echo "Single-threaded decoding"
	PARALLEL=""
fi


# get and print TF version
TF_VERSION=$(python3 -c "import tensorflow as tf ;  print(tf.__version__)")
echo "Your TF version is $TF_VERSION"
echo "TF versions 1.3.0, 1.4.0, 1.5.0 and 1.6.0 are tested"


# compile it for TF1.3
if [ "$TF_VERSION" == "1.3.0" ]; then

	echo "Compiling for TF 1.3.0 now ..."

	TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

	g++ -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 $PARALLEL -I$TF_INC 


# compile it for TF1.4
elif [ "$TF_VERSION" == "1.4.0" ]; then

	echo "Compiling for TF 1.4.0 now ..."
	
	TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
	TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

	g++ -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -D_GLIBCXX_USE_CXX11_ABI=0 $PARALLEL -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework

# all other versions (tested for: TF1.5 and TF1.6)
else
	echo "Compiling for TF 1.5.0 or 1.6.0 now ..."

	TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
	TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


	g++ -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 $PARALLEL

fi
