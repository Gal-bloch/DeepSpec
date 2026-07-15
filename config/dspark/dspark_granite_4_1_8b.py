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
    # Proof-of-concept on 1 GPU: grad-accum = global/(world_size*local) = 16/1.
    # Must stay divisible by world_size*local_batch_size if GPU count changes.
    global_batch_size=16,
    num_train_epochs=3,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=1000,
)

data = dict(
    target_cache_path=None,
    chat_template="granite",
    # Shorter than the 4096 default to keep the target cache within ~1 TB
    # for the proof-of-concept run.
    max_length=2048,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
