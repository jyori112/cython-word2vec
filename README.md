実験結果: https://docs.google.com/spreadsheets/d/1l_OqjuCC2D44KvFFDYHDeN7GRReRS9nh7JXytctfxHw/edit#gid=0

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
