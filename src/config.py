from pydantic import BaseModel


class Config(BaseModel):
    embedding_size = 384
    block_size = 128
    num_heads = 6
    num_blocks = 6
    lr = 1e-5
    wd = 1e-2
    batch_size = 16
    n_epochs = 5000
    device = "mps"
    eval_every = n_epochs // 10
    num_eval_steps = 200
    accumulate_grad = 4
