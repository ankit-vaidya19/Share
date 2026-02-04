# Adapting Flux using Share

Our experiments for Image Generation are run on a single NVIDIA A5000 GPU. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

<h1 align="center"> 
    <image src="../imgs/flux.png"/>
</h1>

## Quick Start

```bash
cd Diffusion
./scripts/run_share_continual.sh [output_dir] [model_name]

# Example:
./scripts/run_share_continual.sh ./outputs black-forest-labs/FLUX.1-dev
```

## Run Manually

```bash
python flux.py \
    --lora_paths "path/to/style1" "path/to/style2" "path/to/style3" \
    --lora_names "style1" "style2" "style3" \
    --prompts "prompt1" "prompt2" "prompt3" \
    --model_name_or_path "black-forest-labs/FLUX.1-dev" \
    --num_components 32 \
    --output_dir "./outputs"
```


```
outputs/
├── T0/
│   └── style1.png
├── T1/
│   ├── style2.png
│   ├── style1_updated.png
│   └── style1_updated.safetensors
└── T2/
    ├── style3.png
    ├── style1_updated.png
    ├── style2_updated.png
    ├── style1_updated.safetensors
    └── style2_updated.safetensors
```
