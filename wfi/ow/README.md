# WF-I Scenario

If you are using your preferred locations for datasets and models rather than the home directory, additionally pass the options of --data_root <your_data_parent_dir> --model_root <your_model_parent_dir>. For example, if you download them in /anon/datasets and /anon2/ssl_saved_model, use --data_root /anon --model_root /anon2.

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
python wfi/ow/wfi-ow.py 
```

This will create ~/ssl_saved_model/wfi-ow-pre<precision>.ckpt and ~/datasets/wfi-ow-test.npz (or <your_model_parent_dir>/ssl_saved_model/wfi-ow-pre<precision>.ckpt and <your_data_parent_dir>/datasets/wfi-ow-test.npz)

## OW testing

Code locates at gandalf/wfi/ow/wfi-ow-eval.py

This code tests the saved model (i.e.,ssl_saved_model/wfi-ow-pre<precision>.ckpt) using 5k, 150k, and 360k (i.e.,back_size) of testing data (i.e., datasets/wfi-ow-test.npz)

Run the code as follows. 

```sh
python wfi/ow/wfi-ow-eval.py --back_size 360000 --model_path /ssl_saved_model/wfi-ow-pre<precision>.ckpt --test_path /datasets/wfi-ow-test.npz
```

This will generate the stdout showing <confidence_threshold>\n<precision>\n<recall>

To test our pretrained model used in the paper, download [model](https://docs.google.com/uc?export=download&id=1wWEL8SFw2Ugk38GrYPADd7R_8q7nHPgI) and extract wfi_ow_paper in ~/ssl_saved_model  (or <your_model_parent_dir>/ssl_saved_model) and download [test set](https://docs.google.com/uc?export=download&id=1HVj9BlUT-SGnAyCFGQP1TI60JrVNKmgg) in ~/datasets  (or <your_data_parent_dir>/datasets). Then, type the following.

```sh
python wfi/ow/wfi-ow-eval.py --back_size 360000
```
