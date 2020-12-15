# WF-S Scenario

## CW experiments (using 5,10,20,50,90 instances)

Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets.

Note that we used different train_epochs for each num_labeled_examples as follows.

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

Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 5 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 10 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 20 --train_epochs 3
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 50 --train_epochs 4
~/gdlf_env/bin/python -u wfs/cw/wfs-cw-awf.py --num_labeled_examples 90 --train_epochs 5
```

Results: 29-35% (5ins), 37-41% (10ins), 42-51% (20ins), 54-59% (50ins), 60-63% (90ins)


## OW training (using 90 instances)

Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A), [GDLF-OW](https://docs.google.com/uc?export=download&id=1aT5fLgRVGKwF_-VhB5px-WEAVsEOxhxD), [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv), and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow.py 
```

This will create ssl_saved_model/wfs-ow-pre<precision>.ckpt and datasets/wfs-ow-test.npz

## OW testing

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow-eval.py --back_size 70000 --model_path ~/ssl_saved_model/wfs-ow-pre<precision>.ckpt --test_path ~/datasets/wfs-ow-awf-gdow.npz
```

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1HAkZQUenNk7TczMt2Y8ohSdn26H3UzFr) and extract wfs_ow_paper in ~/ssl_saved_model and download [test set](https://docs.google.com/uc?export=download&id=1LvYgyrRnspqt8KYqN-yxS5haF0yZ6iuY) in ~/datasets. Then, type the following.

```sh
~/gdlf_env/bin/python -u wfs/ow/wfs-ow-eval.py --back_size 70000
```

## Subpage experiments (using 2,10,20 subpages)

Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 2
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 10
~/gdlf_env/bin/python -u wfs/subpages/wfs-opt.py --num_subpages 20
```

Results: 37-46% (2 subpages), 40-48% (10 subpages), 40-48% (20 subpages)
