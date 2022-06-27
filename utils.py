import os
import json


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
            def opener(path, flags): return os.open(path, flags, mode=0o600)
            with open(user_auth_token, "w", opener=opener) as f:
                json.dump({"username": username, "key": key}, f)


class DummyScaler:
    def __init__(self, *args, **kwargs):
        pass

    def scale(self, loss):
        return loss

    def step(self, opt):
        opt.step()

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
