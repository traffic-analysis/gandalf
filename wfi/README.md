# WF-I Scenario

## CW experiments (using 5,10,20,50,90 instances)

Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA) and [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 5 --train_epochs 12
~/gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 10 --train_epochs 15
~/gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 20 --train_epochs 25
~/gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 50 --train_epochs 20
~/gdlf_env/bin/python -u wfi/cw/wfi-cw.py --num_labeled_examples 90 --train_epochs 20
```

Results: 66-72% (5ins), 79-81% (10ins), 86-88% (20ins), 91-93% (50ins), 94-95% (90ins)


## CW experiments using DF as unlabeled data (using 20 instances)

Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA), and [DF-data](https://docs.google.com/uc?export=download&id=1BEzP3kwtw33BMYp2BKITrG_LFowc1k6b) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfi/cw/wfi-cw20-df.py
```

Results: 85-88%

## OW training (using 20 instances)

Download [AWF1](https://docs.google.com/uc?export=download&id=1nOR_bFdBUn4DAHZdg8Q9N3x5UCBoZGfA), [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R), and [AWF-OW](https://docs.google.com/uc?export=download&id=1K7nr4ReEYMYH04DYOswyxt2Ar_tsJDDS) sets in ~/datasets.

```sh
~/gdlf_env/bin/python -u wfi/ow/wfi-ow.py 
```

This will create ssl_saved_model/wfi-ow-pre<precision>.ckpt and datasets/wfi-ow-test.npz

## OW testing

```sh
~/gdlf_env/bin/python -u wfi/ow/wfi-ow-eval.py --back_size 360000 --model_path ~/ssl_saved_model/wfi-ow-pre<precision>.ckpt --test_path ~/datasets/wfi-ow-test.npz
```

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1wWEL8SFw2Ugk38GrYPADd7R_8q7nHPgI) and extract wfi_ow_paper in ~/ssl_saved_model and download [test set](https://docs.google.com/uc?export=download&id=1HVj9BlUT-SGnAyCFGQP1TI60JrVNKmgg) in ~/datasets. Then, type the following.

```sh
~/gdlf_env/bin/python -u wfi/ow/wfi-ow-eval.py --back_size 360000
```

## Circuit experiments (using 1,5,40 circuits)

Download [DF-circuit](https://docs.google.com/uc?export=download&id=1nb1BvpTYkxWK4Sk6i7iSJhvyQ4fqJPtw), [AWF2](https://docs.google.com/uc?export=download&id=1vbYleGfewHcJqkUsBL9oao7PuPExwl9R), [DF-slow](https://docs.google.com/uc?export=download&id=1Dmc9UaOmb1hRveuQlryHK_4PwzAxugXB), [DF-fast](https://docs.google.com/uc?export=download&id=1F6Qg-VzbMHZNqsMMXR63D-dlKnY08Uj6) in ~/datasets.

### Circuit diversity

```sh
~/gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 1
~/gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 5
~/gdlf_env/bin/python -u wfi/circuit/wfi-circuit-div.py --num_train_circuits 40
```

Results: 84-87% (1 circuit), 85-88% (5 circuits), 86-87% (40 circuits)

### Testing with slow circuits


```sh
~/gdlf_env/bin/python -u wfi/circuit/wfi-circuit-slow.py
```

Results: 91-94%

### Testing with fast circuits

```sh
~/gdlf_env/bin/python -u wfi/circuit/wfi-circuit-fast.py
```

Results: 92-93%
