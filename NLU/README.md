# Adapting RoBERTa using EigenFlux

## Adapting to the GLUE Benchmark

Our experiments on the GLUE benchmark are run on 4 NVIDIA A5000 GPU cards. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds.

<h1 align="center"> 
    <image src="../imgs/GLUE.png"/>
</h1>



## Continual Learning Pipeline

The Share (Continual) method trains on GLUE tasks sequentially, transferring knowledge via EigenFlux adapters:

```
COLA → MRPC → RTE → STSB → QNLI → SST-2
```

### Quick Start: Run Full Pipeline

```bash
cd NLU
./scripts/run_share_continual.sh [output_dir] [source_lora_path] [source_lora_name]

# Example:
./scripts/run_share_continual.sh ./outputs ankit-vaidya19/cola_lora_r_8 cola
```

### Step-by-Step: Run Individual Scripts

**1. Generate EigenFlux initialization:**
```bash
cd NLU
bash scripts/generation/get_eigenflux_cola.sh
```

**2. Train tasks sequentially:**
```bash
bash scripts/training/glue_cola.sh   # Loads: cola_eigenflux → Saves: cola_eigenflux_trained
bash scripts/training/glue_mrpc.sh   # Loads: cola_eigenflux_trained → Saves: mrpc_eigenflux_trained
bash scripts/training/glue_rte.sh    # Loads: mrpc_eigenflux_trained → Saves: rte_eigenflux_trained
bash scripts/training/glue_stsb.sh   # Loads: rte_eigenflux_trained → Saves: stsb_eigenflux_trained
bash scripts/training/glue_qnli.sh   # Loads: stsb_eigenflux_trained → Saves: qnli_eigenflux_trained
bash scripts/training/glue_sst2.sh   # Loads: qnli_eigenflux_trained → Saves: sst2_eigenflux_trained
```


