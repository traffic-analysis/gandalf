# WF-S Scenario

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

## OW training (using 90 instances)

Code locates at gandalf/wfs/cw/wfs-ow.py

This code trains the model using 90 instances (i.e., num_labeled_examples) of 25 monitored sites and 2,250 unmonitored websites in the open-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled (monitored) | GDLF25  | 90 x 25  |
| Labeled (unmonitored)  | GDLF-OW  | 1 x 2250  |
| Unlabeled (monitored)  | AWF1  | 2498 x 100  |
| Unlabeled (unmonitored)  | GDLF-OW-old  | 1 x 83k  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A), [GDLF-OW](https://docs.google.com/uc?export=download&id=1aT5fLgRVGKwF_-VhB5px-WEAVsEOxhxD), [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv), and [GDLF-OW-old](https://docs.google.com/uc?export=download&id=1xLYTNzf1hJMTlvpjvFCBurvo7jmPRwMz) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. Note that we started saving the model when testing precision > 0.79. We reported the result in the paper using the model (precision=0.81-82).

```sh
python wfs/ow/wfs-ow.py 
```

This will create ~/ssl_saved_model/wfs-ow-pre<precision>.ckpt and ~/datasets/wfs-ow-test.npz (or <your_model_parent_dir>/ssl_saved_model/wfs-ow-pre<precision>.ckpt and <your_data_parent_dir>/datasets/wfs-ow-test.npz)

## OW testing

Code locates at gandalf/wfs/cw/wfs-ow-eval.py

This code tests the saved model (i.e.,ssl_saved_model/wfs-ow-pre<precision>.ckpt) using 20k, 50k, and 70k (i.e.,back_size) of testing data (i.e., datasets/wfs-ow-test.npz)

Run the code as follows. 

```sh
python wfs/ow/wfs-ow-eval.py --back_size 70000 --model_path /ssl_saved_model/wfs-ow-pre<precision>.ckpt --test_path /datasets/wfs-ow-awf-gdow.npz
```

This will generate the stdout showing <confidence_threshold>\n<precision>\n<recall>

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1HAkZQUenNk7TczMt2Y8ohSdn26H3UzFr) and extract wfs_ow_paper in ~/ssl_saved_model (or <your_model_parent_dir>/ssl_saved_model) and download [test set](https://docs.google.com/uc?export=download&id=1LvYgyrRnspqt8KYqN-yxS5haF0yZ6iuY) in ~/datasets (or <your_data_parent_dir>/datasets). Then, type the following.

```sh
python wfs/ow/wfs-ow-eval.py --back_size 70000
```
