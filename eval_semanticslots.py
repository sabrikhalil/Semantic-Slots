"""
Evaluate SemanticSlots on a validation set with the Refined, Frozen or Adaptive inference mode.

Videos are processed one at a time and the slot initialization noise is seeded per video
(``seed + index``), so the numbers are exactly reproducible. Metrics are averaged over videos.

    python eval_semanticslots.py --cfg_file config-semanticslots/semanticslots-ytvis.py \
        --ckpt_file semanticslots-ytvis.pth --data_dir /path/to/datasets --mode all
"""
from argparse import ArgumentParser
from pathlib import Path
import time

import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config
from object_centric_bench.util_model import interpolat_argmax_attent

MODES = ["refined", "frozen", "adaptive"]
METRICS = ["ari", "ari_fg", "mbo", "miou"]


@pt.inference_mode()
def evaluate(model, dataload, acc_fn, resolut0, modes, seed, max_videos=None):
    results = {m: {k: [] for k in METRICS + ["ms"]} for m in modes}
    total = min(len(dataload), max_videos or len(dataload))
    for i, batch in enumerate(tqdm.tqdm(dataload, total=total)):
        if i >= total:
            break
        batch = {k: v.cuda() for k, v in batch.items()}
        for mode in modes:
            model.m.mode = mode
            pt.manual_seed(seed + i)  # same slot initialization for every mode

            pt.cuda.synchronize()
            t0 = time.perf_counter()
            with pt.autocast("cuda", enabled=True):
                output = model(batch=batch)
            pt.cuda.synchronize()
            results[mode]["ms"].append((time.perf_counter() - t0) * 1000)

            attent2 = output["attent2"]  # decoder cross-attention, (b,t,n,h,w)
            segment = interpolat_argmax_attent(attent2, size=resolut0).long()
            output["segment2"] = ptnf.one_hot(segment).bool()
            acc = acc_fn(output=output, batch=batch)
            for k in METRICS:
                value, valid = acc[k]
                results[mode][k].append(value[0].item() if valid[0] else np.nan)
    return results


def main(args):
    cfg = Config.fromfile(args.cfg_file)
    cfg.dataset_v.base_dir = Path(args.data_dir)

    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        1,
        shuffle=False,
        # With workers the batches are contiguous, without them channels-last. This changes the
        # fp16 kernels of the backbone a little. num_workers=0 gives the exact numbers of the paper.
        num_workers=0,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    model = build_from_config(cfg.model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)
    model.load(args.ckpt_file, None, verbose=False)
    model = model.cuda().eval()
    model.m.adapt_thresh = args.adapt_thresh
    model.m.adapt_iter = args.adapt_iter

    acc_fn = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    modes = MODES if args.mode == "all" else [args.mode]
    results = evaluate(
        model, dataload_v, acc_fn, cfg.resolut0, modes, args.seed, args.max_videos
    )

    print(f"\n{Path(args.cfg_file).stem}  {args.ckpt_file}")
    print(f"{'mode':<10}" + "".join(f"{k:>8}" for k in METRICS) + f"{'ms/video':>10}")
    for mode in modes:
        r = results[mode]
        line = f"{mode:<10}" + "".join(f"{np.nanmean(r[k]) * 100:>8.2f}" for k in METRICS)
        print(line + f"{np.mean(r['ms']):>10.1f}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="refined", choices=MODES + ["all"])
    parser.add_argument("--adapt_thresh", type=float, default=0.04)
    parser.add_argument("--adapt_iter", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_videos", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
