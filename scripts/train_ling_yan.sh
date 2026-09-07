#!/usr/bin/env bash
# Anima character LoRA for Ling Yan, tuned to fit a 6GB RTX 3060 Laptop.
#
# Trained on anima-base-v1.0 rather than the turbo checkpoint we render with.
# LoRAs transfer from base to the step-distilled turbo, and training against a
# distilled model is the unusual path, not the safe one.
#
# The 6GB-specific flags:
#   --blocks_to_swap        streams transformer blocks between CPU and GPU
#   --qwen_image_vae_2d     ~1/3 the VAE peak VRAM and about 2x faster
#   --gradient_checkpointing trades compute for activation memory
#   --cache_latents / --cache_text_encoder_outputs  keeps the VAE and Qwen3
#       encoder off the GPU for the whole training run, which on a card this
#       size matters more than either flag does on its own
#   adafactor rather than AdamW8bit: no bitsandbytes dependency on Windows and
#       a smaller optimiser state.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="C:/ComfyUI/ComfyUI_windows_portable/ComfyUI/models"
TRAIN="$ROOT/workspace/projects/psionic-cultivation/bible/_training/ling_yan"
OUT="$MODELS/loras"

EXTRA=()
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA+=(--max_train_steps 2 --output_name ling_yan_smoke)
else
  EXTRA+=(--max_train_epochs 8 --save_every_n_epochs 2 --output_name ling_yan_v1)
fi

cd "$ROOT/tools/sd-scripts"
# sd-scripts logs bilingual messages; on a Windows console defaulting to
# cp1252 the Japanese half raises UnicodeEncodeError and kills the run.
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$ROOT/.venv/Scripts/python.exe" -m accelerate.commands.launch \
  --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  anima_train_network.py \
  --pretrained_model_name_or_path "$MODELS/diffusion_models/anima-base-v1.0.safetensors" \
  --qwen3 "$MODELS/text_encoders/qwen_3_06b_base.safetensors" \
  --vae "$MODELS/vae/qwen_image_vae.safetensors" \
  --dataset_config "$TRAIN/dataset.toml" \
  --output_dir "$OUT" \
  --save_model_as safetensors \
  --network_module networks.lora_anima \
  --network_dim 32 \
  --network_alpha 16 \
  --learning_rate 2e-5 \
  --optimizer_type adafactor \
  --lr_scheduler constant \
  --timestep_sampling sigmoid \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --network_train_unet_only \
  --blocks_to_swap 20 \
  --qwen_image_vae_2d \
  --seed 42 \
  "${EXTRA[@]}"
