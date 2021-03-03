# WF-S Scenario

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

## CW experiments (using 5,10,20,50,90 instances)

Code locates at gandalf/wfs/cw/wfs-cw.py

This code trains the model using 5-90 instances (i.e., num_labeled_examples) of 25 sites and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 5-90 x 25  |
| Unlabeled  | GDLF-OW-old  | 1 x 83k  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. Note that we used different number of epochs (i.e.,train_epochs) per num_labeled_examples, which we found as more effective one. 

```sh
python wfs/cw/wfs-cw.py --num_labeled_examples 5 --train_epochs 6
python wfs/cw/wfs-cw.py --num_labeled_examples 10 --train_epochs 10
python wfs/cw/wfs-cw.py --num_labeled_examples 20 --train_epochs 10
python wfs/cw/wfs-cw.py --num_labeled_examples 50 --train_epochs 10
python wfs/cw/wfs-cw.py --num_labeled_examples 90 --train_epochs 10
```

Results: 28-33% (5ins), 35-43% (10ins), 44-52% (20ins), 58-59% (50ins), 58-64% (90ins)


Note that sometimes the training CW accuracy did not reach over 80% at the end of running, which yielded lower testing accuracy. In this case, even though we used more optimal epochs found thru tuning for each num_labeled_examples (as shown above), we recommend to add more epochs.

## CW experiments using AWF1 as unlabeled data (using 5,10,20,50,90 instances)

Code locates at gandalf/wfs/cw/wfs-cw-awf.py

This code trains the model using 5-90 instances (i.e., num_labeled_examples) of 25 sites and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 5-90 x 25  |
| Unlabeled  | AWF1  | 2498 x 100  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. Note that we used different number of epochs (i.e.,train_epochs) per num_labeled_examples, which we found as more effective one. 

```sh
python wfs/cw/wfs-cw-awf.py --num_labeled_examples 5 --train_epochs 3
python wfs/cw/wfs-cw-awf.py --num_labeled_examples 10 --train_epochs 3
python wfs/cw/wfs-cw-awf.py --num_labeled_examples 20 --train_epochs 3
python wfs/cw/wfs-cw-awf.py --num_labeled_examples 50 --train_epochs 4
python wfs/cw/wfs-cw-awf.py --num_labeled_examples 90 --train_epochs 5
```

Results: 29-35% (5ins), 37-41% (10ins), 42-51% (20ins), 54-59% (50ins), 60-63% (90ins)
