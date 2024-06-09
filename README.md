# ICIP-ATTENTION-BASED-FEW-SHOT-DIAGNOSIS-OF-CHEST-X-RAYS-USING-SEMANTIC-SIGNATURES
Official Repository of the paper ATTENTION-BASED FEW-SHOT DIAGNOSIS OF CHEST X-RAYS USING SEMANTIC SIGNATURES

In order to execute the code of proposed method just navigate into the codes directory and then run the following command.

```
python chexpert_proposed_128_1.py > logfile.txt
```

To change the dataset split change the path of x and y for each training, testing and validation as follows:

```
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_val_3.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_val_3.npy"
```
to 

```
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_val_4.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_val_4.npy"
```
If you want to perform experiments on split 4 instead of split 3
