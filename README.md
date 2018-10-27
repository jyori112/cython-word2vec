# cython-word2vec

cython implementation of word2vec

## Install

```
$ git clone https://github.com/jyori112/cython-word2vec
$ cd cython-word2vec
$ ./cythonize.sh; python setup.py develop
```

## Usage

Create dictionary first,

```
$ python main.py build_dic [corpus] [dictionary]
```

Train word embeddings

```
$ python main.py train [corpus] [dictionary] [output]
```
