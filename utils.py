import os
import json
import torch
import weakref


def kaggle_setup():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    user_auth_token = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(user_auth_token):
        choice = input("User authentication token does not exists! Continue? [y/n]").strip().lower()
        while choice not in ["y", "n"]:
            choice = input("User authentication token does not exists! Continue? [y/n]").strip().lower()
        if choice == "y":
            try:
                os.mkdir(kaggle_dir)
            except FileExistsError:
                pass
            print(f"Kaggle directory is located at {kaggle_dir}.")
            username = input("Username: ")
            key = input("Key: ")

            def opener(path, flags):
                return os.open(path, flags, mode=0o600)

            with open(user_auth_token, "w", opener=opener) as f:
                json.dump({"username": username, "key": key}, f)


class Interval:
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def __len__(self):
        return (self.left is not None) + (self.right is not None)

    @property
    def middle(self):
        if len(self) == 2:
            return (self.left + self.right) // 2
        else:
            return None

    def __repr__(self):
        return f"Interval(left={self.left}, right={self.right})"


class BatchSizeTuner:
    def __init__(self, dataloader, device, start_bsz=8, timeout=999):

        x = next(iter(dataloader))
        self.input_shape = x.shape[1:]
        self.dtype = x.dtype
        del x

        self.device = device
        self.max_bsz = 0
        self.complete = False
        self.intv = Interval(left=start_bsz)
        self.curr_bsz = start_bsz

        self.timeout = timeout
        self.timer = 0

    def __iter__(self):
        while not self.complete:
            yield 0.1 * torch.randn(
                (self.curr_bsz, *self.input_shape), device=self.device, dtype=self.dtype)
        if self.max_bsz:
            print(f"Optimal batch size is {self.max_bsz}.")
        else:
            print(f"Timeout! Cannot find optimal batch size in {self.timeout} iterations!")
            print(f"Tentative solution is: {self.intv.left} (note: this will not be applied to current run).")

    def update(self, passed: bool):
        if len(self.intv) - 1:
            if passed:
                self.intv.left = self.curr_bsz
            else:
                self.intv.right = self.curr_bsz
            self.curr_bsz = self.intv.middle
            if (self.intv.right - self.intv.left) <= 1:
                self.complete = True
                self.max_bsz = self.intv.left
        else:
            if passed:
                self.intv.left = self.curr_bsz
                self.curr_bsz *= 2
            else:
                if self.curr_bsz <= self.intv.left:
                    self.intv.left, self.intv.right = None, self.curr_bsz
                else:
                    self.intv.right = self.curr_bsz
                self.curr_bsz //= 2

        self.timer += 1
        if self.timer >= self.timeout:
            self.complete = True


class EMA:
    """
    exponential moving average
    inspired by:
    [1] https://github.com/fadel/pytorch_ema
    [2] https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L281-L685
    """

    def __init__(self, model, decay=0.9999):
        shadow = []
        refs = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                shadow.append((k, v.detach().clone()))
                refs.append((k, weakref.ref(v)))
        self.shadow = dict(shadow)
        self._refs = dict(refs)
        self.decay = decay
        self.num_updates = 0
        self.backup = None

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, _ref in self._refs.items():
            assert _ref() is not None, "referenced object no longer exists!"
            self.shadow[k] += (1 - decay) * (_ref().data - self.shadow[k])

    def apply(self):
        self.backup = dict([
            (k, _ref().detach().clone()) for k, _ref in self._refs.items()])
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.shadow[k])

    def restore(self):
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.backup[k])
        self.backup = None

    def __enter__(self):
        self.apply()

    def __exit__(self, *exc):
        self.restore()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "num_updates": self.num_updates
        }
