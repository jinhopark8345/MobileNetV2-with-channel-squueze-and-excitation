# Further Optimize MobileNetV2 with channel squeeze and excitation

# Environment
```bash
Python 3.8
torch 1.10.0+cu113
```


# Quick Start
### Training from scratch
```bash
# Train a network from beginning
python train.py --data_dir <cifar10-dataset-directory-path> \
                --epochs 200                                \
                --batch_size 16                             \
                --summary True

# example
python train.py --data_dir "/home/jinho/ML-datasets/cifar" \
                --epochs 200                               \
                --batch_size 16                            \
                --summary True
```

### Resume Training
```bash


```

### Testing with saved model
```bash
python test.py --data_dir "/home/jinho/ML-datasets/cifar"  \
                --saved_model <check_point_path>           \
                --batch_size 16                            \
                --summary False

# example
python test.py --data_dir "/home/jinho/ML-datasets/cifar"  \
            --saved_model "/home/jinho/Projects/mobilenetv2-with-cse/saved/2022-04-26-00h-15m-51s-batch_size-16-max_epoch-200/epoch-8-acc-8186.pth"  \
            --batch_size 16  \
            --summary False

```

### Project Template I used
[pytorch-template](https://github.com/victoresque/pytorch-template)


### TODO

```
### organize source code
- [ ] add requirements.txt
- [ ] Trainer class : make methods for one epoch running and print statistics
- [ ] Tester class  : make methods for one epoch running and print statistics
- [ ] Trainer, tester class : use dataclass to make code cleaner
- [ ] add data loader for cifar100, imagenette, imagewolf
- [ ] make logger class (check out the template project) / measure inferencing time, size ..

### readme 
- [ ] add more introduction
- [ ] add reference & citation
- [ ] add notion explanation 
- [ ] add environment 
```
