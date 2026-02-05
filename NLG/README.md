# NLG Experiments with Share

## Mistral Instruction-Following with EigenFlux

Our experiments are run on NVIDIA A5000 GPU cards. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

<h1 align="center"> 
    <image src="../imgs/Lots_of_Loras.png"/>
</h1>

## Quick Start

Run the continual learning pipeline:

```bash
cd NLG
./scripts/run_cl_pipeline.sh [output_dir]

# Example:
./scripts/run_cl_pipeline.sh ./cl_outputs
```

## Continual Learning Pipeline

The pipeline processes 490 train LoRAs in batches of 50 (~10 timesteps):

1. **Batch Processing**: At each timestep T, loads 50 new LoRAs
2. **Reconstruction Loading**: For T > 1, loads eigenfluxes from T-1, reconstructs, and combines with current batch
3. **Eigenvector Computation**: Computes new eigenvectors (rank updates = False)
4. **Save Everything**: Saves eigenvectors + eigenfluxes for ALL LoRAs seen so far
5. **Evaluation**: Computes eigenfluxes for train_subset (10) + eval (10) LoRAs

### Output Structure

```
cl_outputs/
├── T1/
│   ├── eigenvectors.safetensors     # Components at T1
│   ├── task769/                      # First batch LoRA
│   │   └── adapter_model.safetensors
│   ├── task664/                      # train_subset LoRA  
│   │   └── adapter_model.safetensors
│   ├── task280/                      # eval LoRA
│   │   └── adapter_model.safetensors
│   └── ...
├── T2/
│   ├── eigenvectors.safetensors
│   ├── task769/                      # Updated from T1
│   ├── task664/                      # Updated from T1
│   ├── task1448/                     # New at T2
│   └── ...
└── T10/
    └── ...
```

## Run Manually

```bash
python get_eigenflux.py \
    --train_json train.json \
    --train_subset_json train_subset.json \
    --eval_json eval.json \
    --batch_size 50 \
    --num_components 256 \
    --base_path /mnt/data1/ankit/Lots_of_LoRAs/iid/ \
    --prefix "Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-" \
    --output_dir ./cl_outputs
```

