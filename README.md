# Neuro-AL

Repository for the Neural Argumentative Learning (NAL): a architecture that integrates
Assumption-Based Argumentation (ABA) with deep learning for image analysis. 

## Setup & Cleanup

1. **Configure the Dataset**
   - Edit the configuration file at `src/[dataset]_config.yaml` to specify the dataset directory, select the classification task, or configure model parameters.
   
2. **Mount Datasets**
   - Create a `.env` file and specify the paths to the CLEVR and CLEVR-Hans datasets.

3. **Initialize and Build**
   - Initialize submodules and build the base container by running:
     ```sh
     ./script.sh --setup
     ```

4. **Cleanup Workspace**
   - To clean up the workspace and stop running containers, run:
     ```sh
     ./script.sh --cleanup
     ```

## Running Neuro-AL

### CLEVR-Hans
Run Neuro-AL on CLEVR-Hans tasks:
```sh
./script.sh --clevr <image_path>
```
**Example:**
```sh
./script.sh --clevr /mnt/CLEVR_Hans/test/images/CLEVR_Hans_classid_2_000737.png
```

### SHAPES
Run Neuro-AL on SHAPES tasks:
```sh
./script.sh --shapes
```

## Training Neuro-AL

### CLEVR
Train a new Neuro-AL model on CLEVR:
```sh
./script.sh --train_clevr
```

### SHAPES
Train a Neuro-AL model on SHAPES:
```sh
./script.sh --train_shapes
```

## Evaluating Neuro-AL

### CLEVR
Evaluate the Neuro-AL model on CLEVR-Hans tasks:
```sh
./script.sh --eval_clevr
```

### SHAPES
Evaluate the Neuro-AL model on SHAPES tasks:
```sh
./script.sh --eval_shapes
```

