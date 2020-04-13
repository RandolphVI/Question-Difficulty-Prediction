# Usage-PyTorch

## Options

### Input and output options

```
  --train-file              STR    Training file.      		Default is `data/Train_sample.json`.
  --validation-file         STR    Validation file.      	Default is `data/Validation_sample.json`.
  --test-file               STR    Testing file.       		Default is `data/Test_sample.json`.
  --word2vec-file           STR    Word2vec model file.		Default is `data/word2vec_300.txt`.
```

### Model option

```
  --pad-seq-len             LIST    Padding Sequence length of data.        Depends on data.
  --embedding-type          INT     The embedding type.                     Default is 1.
  --embedding-dim           INT     Dim of character embedding.             Default is 300.
  --filter-sizes            LIST    Filter sizes.                           Default is [3,3].
  --num-filters             LIST    Number of filters per filter size.      Default is [200,400].  
  --pooling-size            INT     Pooling size.                           Default is 3.
  --lstm-dim                INT     Dim of LSTM neurons.                    Default is 256.
  --lstm-layers             INT     Number of LSTM layers.                  Defatul is 1.
  --attention-type          STR     The attention type.                     Default is 'normal'.
  --attention-dim           INT     Dim of Attention neurons.               Default is 200.
  --fc-dim                  INT     Dim of FC neurons.                      Default is 512.
  --dropout-rate            FLOAT   Dropout keep probability.               Default is 0.5.
```

### Training option

```
  --epochs                  INT     Number of epochs.                       Default is 30.
  --batch-size              INT     Batch size.                             Default is 128.
  --learning-rate           FLOAT   Adam learning rate.                     Default is 0.001.
  --decay-rate              FLOAT   Rate of decay for learning rate.        Default is 0.95.
  --decay-steps             INT     How many steps before decy lr.          Default is 500.
  --evaluate-steps          INT     How many steps to evluate val set.      Default is 500.
  --l2-lambda               FLOAT   L2 regularization lambda.               Default is 0.0.
  --checkpoint-steps        INT     How many steps to save model.           Default is 500.
  --num-checkpoints         INT     Number of checkpoints to store.         Default is 10.
```

## Training

The following commands train a model. (Use TARNN for example)

```bash
$ python3 train_tarnn.py
```

Training a model for a 10 epochs and set batch size as 128.

```bash
$ python3 train_tarnn.py --epochs 10 --batch-size 128
```

In the beginning, you will see the program shows:

![](https://live.staticflickr.com/65535/49767412868_ca51f1eb17_o.png)

**You need to choose Training or Restore. (T for Training and R for Restore)**

After training, you will get the `/log` and  `/run` folder.

- `/log` folder saves the log info file.
- `/run` folder saves the checkpoints.

It should be like this:

```text
.
├── logs
├── runs
│   └── 1586759461 [a 10-digital format]
│       ├── bestcheckpoints
│       ├── checkpoints
│       ├── embedding
│       └── summaries
├── test_tarnn.py
├── text_tarnn.py
└── train_tarnn.py
```

**The programs name and identify the model by using the asctime (It should be 10-digital number, like 1586759461).** 

## Restore

When your model stops training for some reason and you want to restore training, you can:

In the beginning, you will see the program shows:

![](https://live.staticflickr.com/65535/49767947506_cbcc0ecfd1_o.png)

**And you need to input R for restore.**

Then you will be asked to give the model name (a 10-digital format, like 1586759461):

![](https://live.staticflickr.com/65535/49767968391_247d21d0bb_o.png)

And the model will continue training from the last time.

## Test

The following commands test a model.

```bash
$ python3 test_tarnn.py
```

Then you will be asked to give the model name (a 10-digital format, like 1586759461):

![](https://live.staticflickr.com/65535/49767454533_6af8053c5f_o.png)

And you can choose to use the best model or the latest model **(B for Best, L for Latest)**:

![](https://live.staticflickr.com/65535/49768319867_0a9fc9cafd_o.png)

Finally, you can get the `predictions.json` file under the `/outputs`  folder, it should be like:

```text
.
├── graph
├── logs
├── output
│   └── 1586759461
│       └── predictions.json
├── runs
│   └── 1586759461
│       ├── bestcheckpoints
│       ├── checkpoints
│       ├── embedding
│       └── summaries
├── test_tarnn.py
├── text_tarnn.py
└── train_tarnn.py
```

