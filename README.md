# Neuro-AL

Repository for the Neural Argumentative Learning (NAL): a architecture that integrates
Assumption-Based Argumentation (ABA) with deep learning for image analysis. 

## Setup

Edit the configuration file at `src/[dataset]_config.yaml` to specify the dataset directory, select the classification task or configure model parameters

Initialize the submodules and build the base container by running:
   ```sh
   ./script.sh --setup
   ```

## Running Neuro-AL

### Using CLEVR-Hans
To run Neuro-AL on CLEVR-Hans tasks, use:
```sh
./script.sh --clevr <image_path>
```

### Using SHAPES
To run Neuro-AL on SHAPES tasks, use:
```sh
./script.sh --shapes
```



