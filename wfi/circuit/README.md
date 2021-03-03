# WF-I Scenario

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

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
python wfi/circuit/wfi-circuit-div.py --num_train_circuits 1
python wfi/circuit/wfi-circuit-div.py --num_train_circuits 5
python wfi/circuit/wfi-circuit-div.py --num_train_circuits 40
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
python wfi/circuit/wfi-circuit-slow.py
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
python wfi/circuit/wfi-circuit-fast.py
```

Results: 92-93%
