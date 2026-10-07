# Notes for coding agents

This repo is the code of *Semantic Slots for Video Object-Centric Learning* (BMVC 2026), a fork of
[RandSF.Q](https://github.com/Genera1Z/RandSF.Q).

## Where things are

- `object_centric_bench/model/semantic_slots.py`: everything specific to SemanticSlots
  (`SemanticSlots` model with the refined, frozen and adaptive modes, `ARTransformerDecoder`, `DINOViT`).
- `config-semanticslots/semanticslots-{ytvis,movi_c,movi_d}.py`: the settings used for the paper.
- `eval_semanticslots.py`: per-video evaluation of the three inference modes.
- `train.py`: RandSF.Q training script. Configs are plain Python files, there are no CLI overrides.
- Everything else is RandSF.Q. Lines we changed there are tagged `[SemanticSlots]`.

## Commands

```shell
pip install -r requirements.txt
python train.py --seed 42 --cfg_file config-semanticslots/semanticslots-ytvis.py --data_dir DATA --save_dir save
python eval_semanticslots.py --cfg_file config-semanticslots/semanticslots-ytvis.py --ckpt_file CKPT --data_dir DATA --mode all
```

`DATA` holds `ytvis/`, `movi_c/` and `movi_d/`, each with `train.lmdb` and `val.lmdb` (links in README.md).

## Checks after a change

1. With the YTVIS checkpoint, `eval_semanticslots.py --mode all` must print
   refined 86.89 / 77.45 / 62.10 / 59.51, frozen 86.39 / 76.66 / 60.83 / 58.26 and
   adaptive 86.64 / 77.09 / 61.26 / 58.57 (ARI / ARI_fg / mBO / mIoU). These are Table 2 of the paper.
   `--max_videos 20` is a fast smoke test.
2. RandSF.Q, SlotContrast and VideoSAUR configs must keep their behavior. When you add an option to a
   RandSF.Q class, its default must reproduce the original code.
3. A short `train.py` run must work: copy a config, set `total_step = 60`, `val_interval = 30`, `batch_size_t = 16`.

## Things that are easy to get wrong

- YTVIS slots come from `NormalShared`, which samples noise in eval mode too. Seed before each forward pass
  to reproduce numbers. MOVi slots come from an `MLP` on the first-frame object boxes (`batch.bbox`, padded
  to 11 or 21 by `ClPadTo1`), as in the RandSF.Q `_c` configs. This is deterministic.
- MOVi validation batches of 16 videos need more than 16 GB of GPU memory. Lower `batch_size_v` if needed,
  it does not change the metrics.
- Keep `num_workers=0` in `eval_semanticslots.py`. With workers, batches are contiguous instead of
  channels-last, the backbone picks other fp16 kernels, and results move by about 0.01 mBO per video.
- Masks come from the decoder cross-attention (`attent2`), not from Slot Attention (`attent`).
  In frozen and adaptive modes, `attent` is zero on frames where Slot Attention does not run.
- Training uses one frame per video (`StridedRandomSlice1`, `size=1`) and only the reconstruction loss.
- `trunc_bp="bi-level"`, AdamW and 4 decoder heads are part of the recipe. Changing them changes results.
- The paper checkpoints hold keys of an older decoder and backbone wrapper. `_load_from_state_dict` in
  `semantic_slots.py` maps them, so old and new checkpoints both load strictly.
- `DINOViT` (MOVi) downloads `facebookresearch/dino` through `torch.hub` on first use.
