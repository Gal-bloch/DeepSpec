import os

from deepspec.trainer import GraniteDSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, GRANITE_4_1_8B

project_name = "deepspec"
exp_name = "dspark_block7_granite_4_1_8b"
seed = 42

model = dict(
    target_model_name_or_path=GRANITE_4_1_8B,
    block_size=7,
    num_draft_layers=5,
    # granite-4.1-8b has 40 hidden layers; spread anchors across depth
    # (mirrors the Qwen3-4B [1,9,17,25,33] spread over 36 layers).
    target_layer_ids=[2, 11, 20, 29, 38],
    # Repurpose Granite's <|unused_1|> token (id 100266) as the DSpark mask token.
    mask_token_id=100266,
    num_anchors=512,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=GraniteDSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Full 8-GPU / full-data run matches the DSpark paper's own setup, so these
    # revert to the proven Qwen3 defaults. global_batch_size must stay divisible by
    # world_size*local_batch_size (base_trainer._compute_gradient_accumulation_steps):
    # on 8 GPUs with local_batch_size=1, 512 -> grad-accum 64.
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    # no_shard is the only strategy exercised by any repo config; the custom
    # BF16Optimizer is not FSDP-shard-safe. The draft is small (5 layers) so
    # no_shard's replicated optimizer state is fine on a large-memory node.
    sharding_strategy="no_shard",
    # DSpark default (proven recipe). flex_attention needs the inductor/Triton build
    # toolchain; that is an ENVIRONMENT requirement on the cluster, not a code knob.
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="granite",
    # Full DSpark default. NOTE: drives target-cache size (tens of TB for full data).
    max_length=4096,
    # 0 is deliberate: persistent_workers=True is hardcoded in base_trainer and the
    # CUDAPrefetcher + persistent-workers combo deadlocked in the PoC. A dataloader
    # hang on a multi-day 8-GPU job wastes the whole allocation; 0 is the proven-safe
    # value. Raise only if the smoke test proves >0 is stable at scale.
    num_workers=0,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
