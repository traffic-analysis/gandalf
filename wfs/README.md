# WF-S Scenario

## CW experiments (using 5,10,20,50,90 instances)

Code locates at ~/gandalf/wfs/cw/wfs-cw.py

This code trains the model using 5-90 instances (i.e., num_labeled_examples) of 25 sites and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 5-90 x 25  |
| Unlabeled  | GDLF-OW-old  | 1 x 83k  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets.

2. Run the code as follows. Note that we used different number of epochs (i.e.,train_epochs) per num_labeled_examples, which we found as more effective one. 

```sh
~/gdlf_env/bin/python -u wfs/cw/wfs-cw.py --num_labeled_examples 5 --train_epochs 6
~/gdlf_env/bin/python -u wfs/cw/wfs-cw.py --num_labeled_examples 10 --train_epochs 10
~/gdlf_env/bin/python -u wfs/cw/wfs-cw.py --num_labeled_examples 20 --train_epochs 10
~/gdlf_env/bin/python -u wfs/cw/wfs-cw.py --num_labeled_examples 50 --train_epochs 10
~/gdlf_env/bin/python -u wfs/cw/wfs-cw.py --num_labeled_examples 90 --train_epochs 10
```

Results: 28-33% (5ins), 35-43% (10ins), 44-52% (20ins), 58-59% (50ins), 58-64% (90ins)


Note that sometimes the training CW accuracy did not reach over 80% at the end of running, which yielded lower testing accuracy. In this case, even though we used more optimal epochs found thru tuning for each num_labeled_examples (as shown above), we recommend to add more epochs.

## CW experiments using AWF1 as unlabeled data (using 5,10,20,50,90 instances)

Code locates at ~/gandalf/wfs/cw/wfs-cw-awf.py

This code trains the model using 5-90 instances (i.e., num_labeled_examples) of 25 sites and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 5-90 x 25  |
| Unlabeled  | AWF1  | 2498 x 100  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets.

2. Run the code as follows. Note that we used different number of epochs (i.e.,train_epochs) per num_labeled_examples, which we found as more effective one. 

```sh
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 5 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 10 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 20 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 50 --train_epochs 4
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 90 --train_epochs 5
```

Results: 29-35% (5ins), 37-41% (10ins), 42-51% (20ins), 54-59% (50ins), 60-63% (90ins)


## OW training (using 90 instances)

Code locates at ~/gandalf/wfs/cw/wfs-ow.py

This code trains the model using 90 instances (i.e., num_labeled_examples) of 25 monitored sites and 2,250 unmonitored websites in the open-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled (monitored) | GDLF25  | 90 x 25  |
| Labeled (unmonitored)  | GDLF-OW  | 1 x 2250  |
| Unlabeled (monitored)  | AWF1  | 2498 x 100  |
| Unlabeled (unmonitored)  | GDLF-OW-old  | 1 x 83k  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A), [GDLF-OW](https://docs.google.com/uc?export=download&id=1aT5fLgRVGKwF_-VhB5px-WEAVsEOxhxD), [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv), and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets.

2. Run the code as follows. Note that we started saving the model when testing precision > 0.79. We reported the result in the paper using the model (precision=0.81-82).

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow.py 
```

This will create ~/ssl_saved_model/wfs-ow-pre<precision>.ckpt and ~/datasets/wfs-ow-test.npz

## OW testing

Code locates at ~/gandalf/wfs/cw/wfs-ow-eval.py

This code tests the saved model (i.e.,~/ssl_saved_model/wfs-ow-pre<precision>.ckpt) using 20k, 50k, and 70k (i.e.,back_size) of testing data (i.e., ~/datasets/wfs-ow-test.npz)

Run the code as follows. 

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow-eval.py --back_size 70000 --model_path ~/ssl_saved_model/wfs-ow-pre<precision>.ckpt --test_path ~/datasets/wfs-ow-awf-gdow.npz
```

This will generate the stdout showing <confidence_threshold>\n<precision>\n<recall>

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1HAkZQUenNk7TczMt2Y8ohSdn26H3UzFr) and extract wfs_ow_paper in ~/ssl_saved_model and download [test set](https://docs.google.com/uc?export=download&id=1LvYgyrRnspqt8KYqN-yxS5haF0yZ6iuY) in ~/datasets. Then, type the following.

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow-eval.py --back_size 70000
```

## Subpage experiments (using 2,10,20 subpages)

Code locates at ~/gandalf/wfs/cw/wfs-opt.py

This code trains the model using 20 instances of 25 sites based on different number of training subpages (i.e.,num_subpages) and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 480 x 25  |
| Unlabeled  | AWF1  | 2500 x 100  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets.

2. Run the code as follows.

```sh
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 2
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 10
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 20
```

Results: 37-46% (2 subpages), 40-48% (10 subpages), 40-48% (20 subpages)
