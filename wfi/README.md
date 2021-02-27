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
gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 5 --train_epochs 12
gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 10 --train_epochs 15
gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 20 --train_epochs 25
gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 50 --train_epochs 20
gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 90 --train_epochs 20
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
gdlf_env/bin/python -u wfi/cw/wfi-cw20-df.py
```

Results: 85-88%

## OW training (using 20 instances)

Code locates at gandalf/wfi/ow/wfi-ow.py

This code trains the model using 20 instances (i.e., num_labeled_examples) of 100 monitored sites and 2,000 unmonitored sites in the open-world setting. It saves the trained model and testing set. 

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled (Monitored) | AWF1  | 20 x 100  |
| Labeled (Unmonitored) | AWF-OW  | 1 x 2000  |
| Unabeled (Monitored) | AWF2  | 2500 x 100  |
| Unlabeled (Unmonitored) | AWF-OW  | 1 x 35k  |

To execute the code, 

1. Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA), [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R), and [AWF-OW](https://docs.google.com/uc?export=download&id=1K7nr4ReEYMYH04DYOswyxt2Ar_tsJDDS) sets in ~/datasets  (or <your_data_parent_dir>/datasets).

2. Run the code as follows. Note that we started saving the model when testing precision > 0.39. Although we reported the result in the paper using the model (precision=0.49), we also noticed that this result can be improved using more epochs to yield better precision up to 0.55.

```sh
gdlf_env/bin/python -u wfi/ow/wfi-ow.py 
```

This will create ~/ssl_saved_model/wfi-ow-pre<precision>.ckpt and ~/datasets/wfi-ow-test.npz (or <your_model_parent_dir>/ssl_saved_model/wfi-ow-pre<precision>.ckpt and <your_data_parent_dir>/datasets/wfi-ow-test.npz)

## OW testing

Code locates at gandalf/wfi/ow/wfi-ow-eval.py

This code tests the saved model (i.e.,ssl_saved_model/wfi-ow-pre<precision>.ckpt) using 5k, 150k, and 360k (i.e.,back_size) of testing data (i.e., datasets/wfi-ow-test.npz)

Run the code as follows. 

```sh
gdlf_env/bin/python -u wfi/ow/wfi-ow-eval.py --back_size 360000 --model_path /ssl_saved_model/wfi-ow-pre<precision>.ckpt --test_path /datasets/wfi-ow-test.npz
```

This will generate the stdout showing <confidence_threshold>\n<precision>\n<recall>

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1wWEL8SFw2Ugk38GrYPADd7R_8q7nHPgI) and extract wfi_ow_paper in ~/ssl_saved_model  (or <your_model_parent_dir>/ssl_saved_model) and download [test set](https://docs.google.com/uc?export=download&id=1HVj9BlUT-SGnAyCFGQP1TI60JrVNKmgg) in ~/datasets  (or <your_data_parent_dir>/datasets). Then, type the following.

```sh
gdlf_env/bin/python -u wfi/ow/wfi-ow-eval.py --back_size 360000
```

## Circuit experiments (using 1,5,40 circuits)

Code locates at gandalf/wfi/circuit/wfi-circuit-div.py

This code trains the model using 25 instances of 95 sites collected by different number of circuits (i.e.,num_train_circuits) and tests the model using 100 instances of 95 sites collected by all circuits in the closed-world setting.

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | DF  | 25 x 95  |
| Unlabeled  | AWF2  | 2500 x 100  |

To execute the code, 

1. Download [DF-circuit](https://docs.google.com/uc?export=download&id=1nb1BvpTYkxWK4Sk6i7iSJhvyQ4fqJPtw), [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R), [DF-slow](https://docs.google.com/uc?export=download&id=1Dmc9UaOmb1hRveuQlryHK_4PwzAxugXB), [DF-fast](https://docs.google.com/uc?export=download&id=1F6Qg-VzbMHZNqsMMXR63D-dlKnY08Uj6) in ~/datasets (or <your_data_parent_dir>/datasets).

2. Run the code as follows. 

### Circuit diversity

```sh
gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 1
gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 5
gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 40
```

Results: 84-87% (1 circuit), 85-88% (5 circuits), 86-87% (40 circuits)

### Testing with slow circuits

Code locates at gandalf/wfi/circuit/wfi-circuit-slow.py

This code trains the model using 90 instances of 95 sites collected by all circuits except 4 slow circuits and tests the model using 100 instances of 95 sites collected by 4 slow circuits (i.e.,DF-slow) in the closed-world setting.

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | DF  | 90 x 95  |
| Unlabeled  | AWF2  | 2500 x 100  |

To execute the code, 

```sh
gdlf_env/bin/python -u wfi/circuit/wfi-circuit-slow.py
```

Results: 91-94%

### Testing with fast circuits

Code locates at gandalf/wfi/circuit/wfi-circuit-fast.py

This code trains the model using 90 instances of 95 sites collected by all circuits except 4 fast circuits and tests the model using 100 instances of 95 sites collected by 4 fast circuits (i.e.,DF-fast) in the closed-world setting.

| Data  | Set | Size |
| ------------- | ------------- | ------------- |
| Labeled  | DF  | 90 x 95  |
| Unlabeled  | AWF2  | 2500 x 100  |

To execute the code, 

```sh
gdlf_env/bin/python -u wfi/circuit/wfi-circuit-fast.py
```

Results: 92-93%
