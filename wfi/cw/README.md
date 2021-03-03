# WF-I Scenario

## CW experiments (using 5,10,20,50,90 instances)

Code locates at /gandalf/wfi/cw/wfi-cw.py

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

This code trains the model using 5-90 instances (i.e., num_labeled_examples) of 100 sites and tests the model using 417 testing instances of 100 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | AWF1  | 5-90 x 100  |
| Unlabeled  | AWF2  | 2500 x 100  |

To execute the code, 

1. Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA) and [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. Note that we used different number of epochs (i.e.,train_epochs) per num_labeled_examples, which we found as more effective one. 

```sh
python wfi/cw/wfi-cw.py --num_labeled_examples 5 --train_epochs 12
python wfi/cw/wfi-cw.py --num_labeled_examples 10 --train_epochs 15
python wfi/cw/wfi-cw.py --num_labeled_examples 20 --train_epochs 25
python wfi/cw/wfi-cw.py --num_labeled_examples 50 --train_epochs 20
python wfi/cw/wfi-cw.py --num_labeled_examples 90 --train_epochs 20
```

Results: 66-72% (5ins), 79-81% (10ins), 86-88% (20ins), 91-93% (50ins), 94-95% (90ins)

## CW experiments using DF as unlabeled data (using 20 instances)

Code locates at gandalf/wfi/cw/wfi-cw20-df.py

This code trains the model using 20 instances (i.e., num_labeled_examples) of 100 sites and tests the model using 417 testing instances of 100 sites in the closed-world setting. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | AWF1  | 20 x 100  |
| Unlabeled  | DF  | 1000 x 95  |

To execute the code, 

1. Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA), and [DF-data](https://docs.google.com/uc?export=download&id=1BEzP3kwtw33BMYp2BKITrG_LFowc1k6b) sets in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. 

```sh
python wfi/cw/wfi-cw20-df.py
```

Results: 85-88%