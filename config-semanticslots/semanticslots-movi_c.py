from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    StridedRandomSlice1,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    MOVi,
    ClPadToMax1,
    ClPadTo1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    AdamW,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    Callback,
    AverageLog,
    SaveBestModel,
)
from object_centric_bench.model import (
    SemanticSlots,
    Sequential,
    Interpolate,
    Identity,
    MLP,
    SlotAttention,
    ARTransformerDecoder,
    DINOViT,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

max_num = 10 + 1  # number of slots: one per object box, padded
resolut0 = [256, 256]
resolut1 = [14, 14]
emb_dim = 256
vfm_dim = 768

total_step = 50000
val_interval = 500
batch_size_t = 64
batch_size_v = 16
num_work = 4
lr = 4e-4

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # train on single frames: one random frame per video
    dict(type=StridedRandomSlice1, keys=["video", "segment"], dim=0, size=1),
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=MOVi,
    data_file="movi_c/train.lmdb",
    extra_keys=["segment", "bbox"],
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type=MOVi,
    data_file="movi_c/val.lmdb",
    extra_keys=["segment", "bbox"],
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=ClPadTo1, keys=["bbox"], dims=[1], num=[max_num]),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

model = dict(
    type=SemanticSlots,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, size=[224, 224], interp="bicubic"),
            dict(type=DINOViT, model_name="dino_vitb16", in_size=224),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP,
        in_dim=vfm_dim,
        dims=[vfm_dim, vfm_dim],
        ln="pre",
        dropout=0.0,
        activation="relu",
        kaiming=True,
    ),
    # slots are initialized from the object boxes of the first frame, as RandSF.Q does on MOVi
    initializ=dict(type=MLP, in_dim=4, dims=[emb_dim, emb_dim]),
    aggregat=dict(
        type=SlotAttention,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp="bi-level",
        epsilon=1e-8,
        activation="relu",
    ),
    decode=dict(
        type=ARTransformerDecoder,
        vfm_dim=vfm_dim,
        slot_dim=emb_dim,
        n_patches=resolut1[0] * resolut1[1],
        d_model=vfm_dim,
        n_heads=4,
        n_blocks=4,
    ),
    mode="refined",  # inference mode: "refined", "frozen" or "adaptive"
)
model_imap = dict(input="batch.video", condit="batch.bbox")
model_omap = ["feature", "slotz", "attent", "attent2", "recon"]
ckpt_map = []  # target<-source
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type=AdamW, params=param_groups, lr=lr, weight_decay=0.01)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=0.05)

loss_fn = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
)
_acc_dict_ = dict(
    map=dict(input="output.segment2", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment", "batch.bbox"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
]
after_forward = [
    dict(  # segmentation = argmax of the decoder cross-attention
        type=Lambda,
        ikeys=[["output.attent2"]],
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment2"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
    dict(type=SaveBestModel, save_dir=..., metric="mbo"),
]
