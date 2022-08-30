# Molecular Point Cloud Generation

## About this Repo

This is the main code base for the problem of molecular point cloud generation


## Experiments

To train the Atom Flow, Use the following command:
```
python train.py --type vert --hidden_dim 32 --block_size 6 --encoder_size 4 --gnn_size 1 --autocast 0 --lr 1e-03 --scheduler_step 3 --scheduler_gamma 0.99 --weight_decay 1e-06 --optimiser Adam --batch_size 128 --epochs 1000 --upload True --upload_interval 5
```


To train the Transformer Coordinates Flow, Use the following command:
```
python train.py --type trans --hidden_dim 128 --autocast 0 --block_size 4 --num_layers 8 --lr 1e-03 --scheduler_step 2 --scheduler_gamma 0.992 --weight_decay 0 --optimiser Adam --batch_size 128 --squeeze 1 --squeeze_step 3 --base invariant --epochs 1000 --upload True --upload_interval 10
```

To train the Transformer Coordinates Flow using QM9-19, Use the following command:
```
python train.py --type trans --hidden_dim 128 --autocast 0 --block_size 4 --num_layers 8 --lr 1e-03 --scheduler_step 2 --scheduler_gamma 0.992 --weight_decay 0 --size_constraint 19 --optimiser Adam --batch_size 128 --squeeze 1 --squeeze_step 3 --base invariant --epochs 1000 --upload True --upload_interval 10
```

To train the EGNN coordinates Flow, Use the following command:
```
python train.py --type coor --hidden_dim 32 --block_size 8 --gnn_size 3 --lr 4e-04 --scheduler_step 4 --scheduler_gamma 0.99 --weight_decay 1e-06 --optimiser Adam --batch_size 128 --base invariant --epochs 1000 --upload True --upload_interval 10
```

To train the coordinates Flow with Neural Spline, Use the following command:
```
python train.py --type spline --num_bins 8 --size_constraint 18 --hidden_dim 128 --autocast 0 --block_size 32 --lr 1e-03 --scheduler_step 5 --scheduler_gamma 0.99 --weight_decay 1e-06 --optimiser Adam --batch_size 128 --base invariant --epochs 1000 --upload True --upload_interval 10
```

