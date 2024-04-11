# Environment Setup
```
numpy
panda
pyspark
pytoolkit
tensorflow-gpu==2.3.0
scikit-learn==0.23.2
```

# Usage
To train and test FROG by yourself, run the following command lines:

```
python main.py --model_version=FROG --num_epochs=10 --mod_split_str=1-128,129-384,385-512 --imp_mod_num=2 --train_path=data/train.csv --val_path=data/val.csv --test_path=data/test.csv
```
