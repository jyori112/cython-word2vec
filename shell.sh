DATA=/data/jsakuma/datasets
ROOT=/data/jsakuma/sym_skipgram

pipenv run python main.py train $DATA/europarl/en $ROOT/europarl/en.dic $ROOT/europarl/en.symmetric \
    --symmetric --workers 16
pipenv run python main.py train $DATA/europarl/en $ROOT/europarl/en.dic $ROOT/europarl/en.no-symmetric \
    --no-symmetric --workers 16

pipenv run python main.py train $DATA/wikipedia/en $ROOT/wikipedia/en.min10.dic $ROOT/wikipedia/en.symmetric \
    --symmetric --workers 16
pipenv run python main.py train $DATA/wikipedia/en $ROOT/wikipedia/en.min10.dic $ROOT/wikipedia/en.no-symmetric \
    --no-symmetric --workers 16
