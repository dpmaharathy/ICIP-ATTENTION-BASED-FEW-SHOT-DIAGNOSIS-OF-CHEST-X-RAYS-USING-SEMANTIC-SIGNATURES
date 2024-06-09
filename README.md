# ICIP-ATTENTION-BASED-FEW-SHOT-DIAGNOSIS-OF-CHEST-X-RAYS-USING-SEMANTIC-SIGNATURES
Official Repository of the paper ATTENTION-BASED FEW-SHOT DIAGNOSIS OF CHEST X-RAYS USING SEMANTIC SIGNATURES

In order to execute the code of proposed method just navigate into the codes directory and then run the following command.

```
python chexpert_pwas_128_split1.py > logfile.txt
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
If you want to perform experiments on split 4 instead of split 3.
Also change the path for model saving according in this line of code.

```
path = "/home/maharathy1/MTP/implementation/protonet_attention_signature_loss/models/split5/"

```
The model configuration can be changed in the for loop as follows:
```
    n_way = 8
    n_support = 5
    n_query = 5

    n_way_val = 3
    n_support_val = 5
    n_query_val = 1

    train_x = Xtrain
    train_y = ytrain

    val_x=Xval
    val_y=yval

    max_epoch = 2
    epoch_size = 500
    temp_str = path + 'chexpert_pwas_split1_128*128_' + str(i)
```
