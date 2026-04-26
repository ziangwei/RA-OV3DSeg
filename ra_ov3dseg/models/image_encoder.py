"""MVP-v1 预留：2D image encoder 封装。"""


class ImageEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode_image(self, image):
        raise NotImplementedError("MVP-v1 will implement CLIP/SigLIP image encoding.")
