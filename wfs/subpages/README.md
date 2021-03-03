# WF-S Scenario

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

## Subpage experiments (using 2,10,20 subpages)

Code locates at gandalf/wfs/cw/wfs-opt.py

This code trains the model using 20 instances of 25 sites based on different number of training subpages (i.e.,num_subpages) and tests the model using 5 testing instances of 96 subpages of 25 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | GDLF25  | 480 x 25  |
| Unlabeled  | AWF1  | 2500 x 100  |

To execute the code, 

1. Download [GDLF25](https://docs.google.com/uc?export=download&id=1p49l9Y0NFqTjIuT-1i3oQFa2UZktKy6A) and [AWF1](https://docs.google.com/uc?export=download&id=1Y7QObZn8H1CBfcncU6bhj6Xmx08FMYSv) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows.

```sh
python wfs/subpages/wfs-opt.py --num_subpages 2
python wfs/subpages/wfs-opt.py --num_subpages 10
python wfs/subpages/wfs-opt.py --num_subpages 20
```

Results: 37-46% (2 subpages), 40-48% (10 subpages), 40-48% (20 subpages)
