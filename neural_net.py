import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.constants import Constants
from utils.helpers import Utils

torch.manual_seed(1337)  # for the sake of reproducibility


class Head(nn.Module):
    """
    one head of self-attention
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.query = nn.Linear(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.value = nn.Linear(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(Constants.BLOCK_SIZE, Constants.BLOCK_SIZE)))

        self.dropout = nn.Dropout(Constants.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # compute attension scores ('affinities')
        # (B,T,C) @ (B,C,T) -> (B,T,T)
        weight = q @ k.transpose(-2, -1) * C**-0.5
        # decoder block, ensures tokens don't communicate with the past
        weight = weight.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)

        # perform  the weighted aggregation of values
        v = self.value(x)  # (B, T, C)
        output = weight @  v  # (B,T,T) @ (B,T,C) -> (B,T,C)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, Constants.NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.dropout = nn.Dropout(Constants.DROPOUT)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    """

    def __init__(self, number_of_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embeddings, 4 * number_of_embeddings),
            nn.ReLU(),
            nn.Linear(4 * number_of_embeddings, number_of_embeddings),
            nn.Dropout(Constants.DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """

    def __init__(self, number_of_embeddings, number_of_heads):
        super().__init__()
        head_size = number_of_embeddings // number_of_heads
        self.self_attention = MultiHeadAttention(number_of_heads, head_size)
        self.feed_forward = FeedForward(number_of_embeddings)
        self.layer_norm1 = nn.LayerNorm(number_of_embeddings)
        self.layer_norm2 = nn.LayerNorm(number_of_embeddings)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, Constants.NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.position_embedding_table = nn.Embedding(
            Constants.BLOCK_SIZE, Constants.NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.blocks = nn.Sequential(
            *[Block(Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, number_of_heads=Constants.NUMBER_OF_HEADS) for _ in range(Constants.NUMBER_OF_LAYERS)]

        )
        # nn.LayerNorm(Constants.NUMBER_OF_EMBEDDING_DIMENSIONS)
        self.layer_norm_final = nn.LayerNorm(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS)  # final layer norm
        self.language_model_head = nn.Linear(
            Constants.NUMBER_OF_EMBEDDING_DIMENSIONS, vocab_size)

    def forward(self, inputs, targets=None):
        # inputs and targets are both (B,T) tensor of integers
        # (Batch, Time, Channels) e.g -> [4, 8, 65]
        B, T = inputs.shape
        token_embedding = self.token_embedding_table(inputs)
        position_embeddings = self.position_embedding_table(torch.arange(
            T, device=Utils.get_device()))  # (T, C) integers from zero to  T-1
        # (B, T, C) x now holds both the token identities and the position ther occur at
        x = token_embedding + position_embeddings
        # x = self.self_attention_head(x)  # apply one head of the self-attention (B,T,C)
        x = self.blocks(x)  # (B, T, C)
        logits = self.language_model_head(x)  # (B, T, vocab_size)

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
            # crop inputs to the last block_size tokens
            inputs_cond = inputs[:, -Constants.BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(inputs_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            inputs_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append samples index to the running sequence
            inputs = torch.cat((inputs, inputs_next), dim=1)  # (B, T + 1)
        return inputs
