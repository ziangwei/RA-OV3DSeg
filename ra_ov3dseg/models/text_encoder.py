"""MVP-v1 预留：text encoder 封装。"""


class TextEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode_text(self, class_names):
        raise NotImplementedError("MVP-v1 will implement text embedding extraction.")
