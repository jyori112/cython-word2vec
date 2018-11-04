DATA=/data/jsakuma/datasets
ROOT=/data/jsakuma/sym_skipgram

pipenv run python main.py train $DATA/europarl/en.sample $ROOT/europarl/en.dic $ROOT/europarl/en.symmetric --symmetric
pipenv run python main.py train $DATA/europarl/en.sample $ROOT/europarl/en.dic $ROOT/europarl/en.no-symmetric --no-symmetric

pipenv run python main.py train $DATA/wikipedia/en.sample $ROOT/wikipedia/en.min10.dic $ROOT/wikipedia/en.symmetric --symmetric
pipenv run python main.py train $DATA/wikipedia/en.sample $ROOT/wikipedia/en.min10.dic $ROOT/wikipedia/en.no-symmetric --no-symmetric
