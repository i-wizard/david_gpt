import torch


class Utils:
    def get_device() -> str:
        """
        Allows to run on GPU if available
        """
        return 'cuda' if torch.cuda.is_available(
        ) else 'cpu'
