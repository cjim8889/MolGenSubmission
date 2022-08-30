# Molecular Graph Generation

## About this Repo

This is the main code base for the problem of molecular graph generation

## Experiments

To run the Argmax Flow Edge Generation Experiment, Use the following command:
```
python train.py --type argmaxadj --batch_size 256 --epochs 200
```


To run the Argmax Flow Edge Generation Experiment V2 with hydrogen atoms removed and triangular tensor, Use the following command:
```
python train.py --type argmaxadjv2 --batch_size 128 --epochs 200 --block_length 6
```

