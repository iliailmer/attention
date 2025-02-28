import json


class Config:
    def __init__(
        self,
        embedding_size=384,
        block_size=128,
        num_heads=6,
        num_blocks=6,
        lr=1e-5,
        wd=1e-2,
        batch_size=16,
        n_epochs=5000,
        device="mps",
        num_eval_steps=200,
        accumulate_grad=4,
    ):
        self.embedding_size = embedding_size
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.eval_every = self.n_epochs // 10
        self.num_eval_steps = num_eval_steps
        self.accumulate_grad = accumulate_grad

    def save(self, fname: str = "config.json"):
        conf_dict = self.__dict__
        with open(fname, "w") as f:
            json.dump(conf_dict, f, indent=4)

    def load(self, fname: str = "config.json"):
        with open(fname, "r") as f:
            conf_dict = json.load(f)
        for k, v in conf_dict.items():
            setattr(self, k, v)
