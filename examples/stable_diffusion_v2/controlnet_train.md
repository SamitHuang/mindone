# ControlNet Training on 910B

## Environment

- Hardware: 910B
- MindSpore: 2.2 20231114

## Train a ControlNet from SD1.5

### 1. Model Weight Conversion

Once the stable diffusion v1.5 model weights (for mindspore)  are saved in `models`, you can run the following command to create a ControlNet init checkpoint. 

```
python tools/sd_add_control.py 
```

### 2. Data Preparation

We will use Fill50k dataset to let the model learn to generate images following the edge control. Download it and put it under `datasets/` folder

For convenience, we take the first 1K samples as training set, which can done by keeping the first 1000 lines in `datasets/fill50k/prompt.json` and removing the rest.


### 3. Training

```
sh scripts/run_train_cldm.sh $CARD_ID
```

The resulting log and checkpoints will be saved in $output_dir as defined in the script.

### 4. Evaluation
To evalute the training result, please run the following script and indicate the path to the trained checkpoint. 

```
sh scripts/run_infer_cldm.sh $CARD_ID $CHECKPOINT_PATH
```

And modify the control image path in the script.


## Finetune an Existing ControlNet
