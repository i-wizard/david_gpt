from typing import List, Tuple

import torch

from neural_net import BigramLanguageModel
from utils.encoders import Encoder


class Transformer:
    def __init__(self):
        self.training_data = ""
        self.encoded_data: List[int] = []
        self.batch_size = 32  # the number of independent sequences will we process in parallel
        self.block_size = 8  # maximum content length for predictions
        self.vocab_size = None
        self.max_iters = 3000  # number of iterations to run training loop on
        self.learning_rate = 1e-2
        self.eval_interval = 300
        self.device = 'cuda' if torch.cuda.is_available(
        ) else 'cpu'  # allows to run on GPU if available
        self.eval_iters = 200

    def load_training_data(self):
        text = ''
        with open("data.txt", 'r') as f:
            text = f.read()
        self.training_data = text

    def tokenize_characters(self):
        characters: List[str] = sorted(list(set(self.training_data)))
        self.vocab_size = len(characters)
        self.encoder = Encoder(characters)
        self.encoded_data = self.encoder.encode(self.training_data)

    def get_batch(self, split: str) -> Tuple[torch.stack, torch.stack]:
        # generate a small batch of data of inputs x and targets y
        data = self.training_data if split == 'train' else self.validation_data
        # generate random 4 numbers between data size and block size
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i: i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1: i + self.block_size + 1]for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        output = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            output[split] = losses.mean()
        self.model.train()
        return output

    def run_training_loop(self):
        for iter in range(self.max_iters):
            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(
                    f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            x_batch, y_batch = self.get_batch('train')

            # evaluate the loss
            logits, loss = self.model(x_batch, y_batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def train(self):
        self.load_training_data()
        self.tokenize_characters()
        tensor = torch.tensor(self.encoded_data, dtype=torch.long)
        # Split up data into training data and validation data
        training_data_upper_boundary: int = int(0.9 * len(tensor))
        self.training_data: torch.tensor = tensor[:training_data_upper_boundary]
        self.validation_data: torch.tensor = tensor[training_data_upper_boundary:]

        # We'll train the model with chunks of data
        torch.manual_seed(1337)
        self.model = BigramLanguageModel(self.vocab_size)
        blm = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)
        self.run_training_loop()
        # create a one by one tensor holding  zeros
        idx = torch.zeros((1, 1), dtype=torch.long)
        print(self.encoder.decode(blm.generate(
            idx, max_new_tokens=500)[0].tolist()))
