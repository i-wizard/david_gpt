import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(1337)  # for the sake of reproducibility


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs and targets are both (B,T) tensor of integers
        # (Batch, Time, Channels) e.g -> [4, 8, 65]
        logits = self.token_embedding_table(inputs)
        loss = None
        # we also want to measure quality of predictions as loss
        if targets is not None:
            Batch, Time, Channels = logits.shape
            # the cross_entropy function expects the logits as a 2 dimensional array
            # which is why we are spreading the B and T into one array
            # the cross_entropy function also expects the target as a one dimensional array
            logits = logits.view(Batch * Time, Channels)  # (B, C)
            targets = targets.view(Batch * Time)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # inputs is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(inputs)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            inputs_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append samples index to the running sequence
            inputs = torch.cat((inputs, inputs_next), dim=1)  # (B, T + 1)
        return inputs
