Java Alpha Zero Reinforcement Learning with deeplearning4j
==========================================================
The provide AlphaZero learning here is a Java implementation of the alpha zero algorithm using deeplearning4j library.

## Introduction
There are already several Alpha[Go] Zero-related projects on github in python and also in C++.
The Java implementation here could help to reuse your existing Java game logics with alpha zero reinforcement learning.
This approach might also be simpler for individuals more familiar with Java than python. 

### Alpha[Go] Zero Algorithm
For nearly 20 years after IBM DeepBlue defeated Kasparov 1997 in chess, 19x19 Go remained a domain where computers were far from achieving human-level play. This changed definitely with AlphaGo's victory over Lee Sedol in the Google Deepmind Challenge Match. Alpha Go has beaten the worlds best Go Player in 5 Games 4:1. There is a very interesting Movie available on Youtube around the event: [AlphaGo - The Movie](https://www.youtube.com/watch?v=WXuK6gekU1Y). Aside the DeepMind efforts it also gives insights into the ideology and philosophy of the Go Game in asian countries.

The AlphaGo algorithm was further adapted and improved with AlphaGo Zero [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) and Alpha Zero [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf).

### Why There Aren't Thousands of Strong Artificial Go intelligences Now ?
It is not enough to know the algorithm and having an implementation ready. A rough calculation of [Computer-Go zero performance](http://web.archive.org/web/20190205013627/http://computer-go.org/pipermail/computer-go/2017-October/010307.html) estimates the training time to generate as much positions as AlphaGo zero could take 1700 years on a single machine with standard hardware.

Efforts like [leela-zero](https://github.com/leela-zero/leela-zero) try to replicate the training process in a public and distributed manner. Similarly [minigo](https://github.com/tensorflow/minigo) attempted to reproduce the learning progress.

It's easier to adapt the algorithm to less complex board games like connect four, Gomoku, Othello and others. With such games it's more realistic to perform the necessary training to obtain a strong artificial intelligence.

## Using Java Alpha Zero
The goal of Java alpha-zero-learning is to enable alpha zero for less complex games. The implementation does not support distributed learning. It is designed to run on a single machine, optionally using graphic cards for net updates.

### Generic release build
You can use the existing Java alpha-zero-learning with the generic published release builds. With ch.evolutionsoft.rl.alphazero.tictactoe-2.0.0-jar-with-dependencies you can directly repeat the training for the Tic Tac Toe prototype. See also the submodule [tic-tac-toe/README.md](./tic-tac-toe/README.md) for a few more information.

ch.evolutionsoft.rl.alphazero.adversary-learning-2.0.0-jar-with-dependencies would let you reuse the general part of the implementation for other board games. The submodule [alpha-zero-adversary-learning/README.md](./alpha-zero-adversary-learning/README.md) contains hints about a new board game implementation.
	

### Rebuild for your hardware
With deeplearning4j you can use CUDA to perform the neural net model computations on a GPU. You would configure it by replacing the following two dependencies with CUDA dependencies matching your available version.

	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native-platform</artifactId>
	</dependency>
	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native</artifactId>
	</dependency>

Also without GPU there is the AVX/AVX2 performance improvement on newer CPU's. Use the operating system dependencies for avx to enable it. The logs will show a warning when you're running the generic build on a avx/avx2 supported CPU:

	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>nd4j-native</artifactId>
		<version>1.0.0-beta7</version>
		<classifier>windows-x86_64-avx2</classifier>
	</dependency>

### Implement new games
Refer to the submodule [alpha-zero-adversary-learning/README.md](alpha-zero-adversary-learning/README.md) to see what's necessary for a new game implementation.

### Mvn build and packaging
The Maven (mvn) builds for each submodule take several minutes and a lot of different system architecture dependencies are packaged into the jar files with dependencies. This is deeplearning4j related and leading to larger distribution packages.
