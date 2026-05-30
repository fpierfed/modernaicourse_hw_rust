#!/usr/bin/env python3
import os
import math
import sys
import json
import torch
from torch.nn import Module, ModuleList, Parameter, Buffer
from huggingface_hub import hf_hub_download


here = os.path.dirname(os.path.abspath(__file__))
repo = 'zkolter/Llama-3.2-1B-Instruct-Simplified'
for filename in [
    'consolidated.00.pth',
    'params.json',
    'tokenizer.model',
    'tokenizer.py',
]:
    if not os.path.exists(filename):
        hf_hub_download(
            repo_id=repo, filename=filename, repo_type='model', local_dir=here
        )
from tokenizer import Tokenizer, Message, ChatFormat


DEVICE = 'mps'


class Linear(Module):
    """Linear layer with no bias term.  The parameters of the layer are stored
    in a .weight Parameter"""

    def __init__(self, in_dim, out_dim):
        """
        Initialize a linear layer without a bias term.

        Inputs:
            in_dim : int - input feature dimension
            out_dim : int - output feature dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.weight = Parameter(torch.empty(out_dim, in_dim))
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply the linear layer to one or more input vectors.

        Input:
            X : torch.Tensor[float] (... x in_dim) - input tensor
        Output:
            torch.Tensor[float] (... x out_dim) - transformed tensor
        """
        ### BEGIN YOUR CODE
        return X @ self.weight.T
        ### END YOUR CODE


class Embedding(Module):
    def __init__(self, num_tokens, dim):
        """
        Initialize an embedding table over a fixed vocabulary.

        Inputs:
            num_tokens : int - vocabulary size
            dim : int - embedding dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        # Note the commnent above regarding self.weights (i.e. We)
        # we store it already transposed.
        self.weight = Parameter(torch.empty(num_tokens, dim))
        ### END YOUR CODE

    def forward(self, Y):
        """
        Look up embeddings for an integer tensor of token ids.

        Input:
            Y : torch.Tensor[int] (...) - tensor of token indices in [0,
                                                                      num_tokens)
        Output:
            torch.Tensor[float] (... x dim) - embedding vectors for each token
                                                                      id
        """
        ### BEGIN YOUR CODE

        # y_dims = list(Y.shape)
        # x_dims = y_dims + [self.weight.shape[0]]

        # Rebuild X from Y and then do the nultiplication
        # This is what you can accomplish with
        # X = F.one_hot(Y, num_classes=self.weight.shape[0]).float()
        # X = torch.zeros(*x_dims)
        # X.scatter_(dim=-1, index=Y.unsqueeze(-1), value=1.0)

        # No need to transpose self.weight: see above.
        # return X @ self.weight

        # Turn out that all of the above can be replace by self.weight[Y]...
        return self.weight[Y]
        ### END YOUR CODE


def silu(x):
    """
    Apply the SiLU nonlinearity elementwise.

    Input:
        x : torch.Tensor[float] (...) - input tensor
    Output:
        torch.Tensor[float] (...) - tensor after applying SiLU
    """
    ### BEGIN YOUR CODE
    return x * torch.sigmoid(x)
    ### END YOUR CODE


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-5):
        """
        Initialize an RMS normalization layer over the last dimension.

        Inputs:
            dim : int - size of the final dimension to normalize
            eps : float - numerical stability constant
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.eps = eps
        self.weight = Parameter(torch.ones(dim))
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply RMS normalization along the final dimension of the input.

        Input:
            X : torch.Tensor[float] (... x dim) - input tensor
        Output:
            torch.Tensor[float] (... x dim) - normalized tensor
        """
        ### BEGIN YOUR CODE
        # Note the keepdim=True, otherwise torch collapses the last dim...
        return (
            X
            * self.weight
            / torch.sqrt(X.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        )
        ### END YOUR CODE


def self_attention(Q, K, V, mask=None):
    """
    Apply scaled dot-product attention, optionally with an additive mask.

    Inputs:
        Q : torch.Tensor[float] (... x query_len x d) - query tensor
        K : torch.Tensor[float] (... x key_len x d) - key tensor
        V : torch.Tensor[float] (... x key_len x d_v) - value tensor
        mask : torch.Tensor[float] (... x query_len x key_len) or None -
                                                                      additive
                                                                      attention
                                                                      mask
    Output:
        torch.Tensor[float] (... x query_len x d_v) - attention output tensor
    """
    ### BEGIN YOUR CODE
    d = Q.shape[-1]

    # Note the K.mT, which is equivalent to K.transpose(-2, -1)
    # meaning only transpose the last 2 dimensions of K
    if mask is not None:
        attn = torch.softmax(Q @ K.mT / math.sqrt(d) + mask, dim=-1) @ V
    else:
        attn = torch.softmax(Q @ K.mT / math.sqrt(d), dim=-1) @ V
    return attn
    ### END YOUR CODE


class MultiHeadAttention(Module):
    def __init__(self, dim, n_heads):
        """
        Initialize a multi-head self-attention layer without caching.

        Inputs:
            dim : int - total embedding dimension
            n_heads : int - number of attention heads
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wp = Linear(dim, dim)
        self.n_heads = n_heads
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Apply multi-head self-attention to a batch of sequences.

        Inputs:
            X : torch.Tensor[float] (batch_size x seq_len x dim) - input
            sequence embeddings
            mask : torch.Tensor[float] (seq_len x seq_len) or None - additive
            attention mask
            seq_pos : int - starting sequence position (unused here)
            use_kv_cache : bool - whether to use KV caching (unused here)
        Output:
            torch.Tensor[float] (batch_size x seq_len x dim) - attention output
        """
        ### BEGIN YOUR CODE
        Q = self.wq(X)
        K = self.wk(X)
        V = self.wv(X)

        batch_size, seq_len, dim = tuple(Q.shape)
        head_dim = dim // self.n_heads
        new_dims = [batch_size, seq_len, self.n_heads, head_dim]

        # Need to use transpose on seq_len, self.n_heads to be able to
        # iterate over the head blocks.
        # To be pedantic:
        # X = torch.tensor([[1, 2, 3, 4, 5, 6],
        #                   [7, 8, 9, 10, 11, 12],
        #                   [13, 14, 15, 16, 17, 18],
        #                   [19, 20, 21, 22, 23, 24],
        #                   [25, 26, 27, 28, 29, 30]])
        # print(X)
        # tensor([[ 1,  2,  3,  4,  5,  6],
        #         [ 7,  8,  9, 10, 11, 12],
        #         [13, 14, 15, 16, 17, 18],
        #         [19, 20, 21, 22, 23, 24],
        #         [25, 26, 27, 28, 29, 30]])
        # print(X.shape)
        # torch.Size([5, 6])
        # X = X.reshape(5, 3, 2).transpose(0, 1)
        # print(X)
        # tensor([[[ 1,  2],
        #          [ 7,  8],
        #          [13, 14],
        #          [19, 20],
        #          [25, 26]],
        #
        #         [[ 3,  4],
        #          [ 9, 10],
        #          [15, 16],
        #          [21, 22],
        #          [27, 28]],
        #
        #         [[ 5,  6],
        #          [11, 12],
        #          [17, 18],
        #          [23, 24],
        #          [29, 30]]])
        #
        # In our case, we just have an extra leading dimension, batch_size
        Q = Q.reshape(new_dims).transpose(1, 2)
        K = K.reshape(new_dims).transpose(1, 2)
        V = V.reshape(new_dims).transpose(1, 2)

        # Process all batches
        Y = self_attention(Q, K, V, mask=mask)

        # Go back to old dimensions, undo all operations in reverse.
        Y = Y.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.wp(Y)
        ### END YOUR CODE


class MultiHeadAttentionKVCache(Module):
    def __init__(self, dim, n_heads, max_cache_size):
        """
        Initialize a multi-head self-attention layer with KV cache buffers.

        Inputs:
            dim : int - total embedding dimension
            n_heads : int - number of attention heads
            max_cache_size : int - maximum sequence length stored in the cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wp = Linear(dim, dim)
        self.n_heads = n_heads

        self.max_cache_size = max_cache_size
        self.k_cache = Buffer(torch.zeros(1, max_cache_size, dim))
        self.v_cache = Buffer(torch.zeros(1, max_cache_size, dim))
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Apply multi-head self-attention, optionally updating and using the KV
        cache.

        Inputs:
            X : torch.Tensor[float] (batch_size x seq_len x dim) - input
            sequence embeddings
            mask : torch.Tensor[float] (seq_len x total_len) or None - additive
            attention mask
            seq_pos : int - starting sequence position for cached tokens
            use_kv_cache : bool - whether to update and use the KV cache
        Output:
            torch.Tensor[float] (batch_size x seq_len x dim) - attention output
        """
        ### BEGIN YOUR CODE
        Q = self.wq(X)
        K = self.wk(X)
        V = self.wv(X)

        if use_kv_cache:
            start, end = seq_pos, seq_pos + X.shape[1]
            self.k_cache[:, start:end, :] = K
            self.v_cache[:, start:end, :] = V
            Working_K = self.k_cache[:, :end, :]
            Working_V = self.v_cache[:, :end, :]
        else:
            Working_K = K
            Working_V = V

        kv_batch_size, kv_seq_len, kv_dim = tuple(Working_K.shape)
        kv_head_dim = kv_dim // self.n_heads
        new_kv_dims = [kv_batch_size, kv_seq_len, self.n_heads, kv_head_dim]

        batch_size, seq_len, dim = tuple(Q.shape)
        head_dim = dim // self.n_heads
        new_q_dims = [batch_size, seq_len, self.n_heads, head_dim]

        # Need to use transpose on seq_len, self.n_heads to be able to
        # iterate over the head blocks.
        Q = Q.reshape(new_q_dims).transpose(1, 2)
        Working_K = Working_K.reshape(new_kv_dims).transpose(1, 2)
        Working_V = Working_V.reshape(new_kv_dims).transpose(1, 2)

        # Process all batches
        Y = self_attention(Q, Working_K, Working_V, mask=mask)

        # Go back to old dimensions, undo all operations in reverse.
        Y = Y.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.wp(Y)
        ### END YOUR CODE


class GatedMLP(Module):
    def __init__(self, dim, ffn_dim):
        """
        Initialize the gated feed-forward network used in the transformer
        block.

        Inputs:
            dim : int - model dimension
            ffn_dim : int - hidden feed-forward dimension
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.w1 = Linear(dim, ffn_dim)
        self.w2 = Linear(ffn_dim, dim)
        self.w3 = Linear(dim, ffn_dim)
        ### END YOUR CODE

    def forward(self, X):
        """
        Apply the gated feed-forward network to the input tensor.

        Input:
            X : torch.Tensor[float] (... x dim) - input tensor
        Output:
            torch.Tensor[float] (... x dim) - transformed tensor
        """
        ### BEGIN YOUR CODE
        return self.w2(silu(self.w1(X)) * self.w3(X))
        ### END YOUR CODE


class TransformerBlock(Module):
    def __init__(self, dim, n_heads, ffn_dim, max_cache_size):
        """
        Initialize a transformer block with attention, normalization, and gated
        MLP layers.

        Inputs:
            dim : int - model dimension
            n_heads : int - number of attention heads
            ffn_dim : int - hidden feed-forward dimension
            max_cache_size : int - maximum sequence length stored in the
            attention cache
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.attn = MultiHeadAttentionKVCache(
            dim=dim, n_heads=n_heads, max_cache_size=max_cache_size
        )
        self.norm1 = RMSNorm(dim=dim)
        self.norm2 = RMSNorm(dim=dim)
        self.mlp = GatedMLP(dim=dim, ffn_dim=ffn_dim)
        ### END YOUR CODE

    def forward(self, X, mask=None, seq_pos=0, use_kv_cache=False):
        """
        Apply one transformer block with residual connections.

        Inputs:
            X : torch.Tensor[float] (batch_size x seq_len x dim) - input
            sequence embeddings
            mask : torch.Tensor[float] (seq_len x total_len) or None - additive
            attention mask
            seq_pos : int - starting sequence position for cached tokens
            use_kv_cache : bool - whether to update and use the attention cache
        Output:
            torch.Tensor[float] (batch_size x seq_len x dim) - transformed
            sequence embeddings
        """
        ### BEGIN YOUR CODE
        Z = X + self.attn(
            self.norm1(X),
            mask=mask,
            seq_pos=seq_pos,
            use_kv_cache=use_kv_cache,
        )
        Y = Z + self.mlp(self.norm2(Z))
        return Y
        ### END YOUR CODE


class Llama3Simplified(Module):
    def __init__(
        self, num_tokens, dim, n_heads, max_seq_len, ffn_dim, num_layers
    ):
        """
        Initialize the simplified Llama 3 model used in this homework.

        Inputs:
            num_tokens : int - vocabulary size
            dim : int - model dimension
            n_heads : int - number of attention heads per layer
            max_seq_len : int - maximum supported sequence length
            ffn_dim : int - hidden feed-forward dimension in each block
            num_layers : int - number of transformer blocks
        """
        super().__init__()
        ### BEGIN YOUR CODE
        self.embedding = Embedding(num_tokens, dim)
        self.pos_embeddings = Parameter(torch.empty(max_seq_len, dim))
        self.layers = ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    ffn_dim,
                    max_cache_size=max_seq_len,  # <-- ??? is that correct ???
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.output = Linear(in_dim=dim, out_dim=num_tokens)
        self.mask = Buffer(
            torch.triu(
                torch.full(
                    size=(max_seq_len, max_seq_len), fill_value=float('-inf')
                ),
                diagonal=1,
            )
        )
        ### END YOUR CODE

    def forward(self, tokens, seq_pos=0, use_kv_cache=False):
        """
        Apply the full language model to a batch of token sequences.

        Inputs:
            tokens : torch.Tensor[int] (batch_size x seq_len) - input token ids
            seq_pos : int - starting sequence position for positional
            embeddings and cached attention
            use_kv_cache : bool - whether to update and use cached keys and
            values
        Output:
            torch.Tensor[float] (batch_size x seq_len x num_tokens) - output
            logits
        """
        ### BEGIN YOUR CODE
        res = self.embedding(tokens)
        # Unsqueeze(0) is needed to ensure broadcast along the batch dim.
        res += self.pos_embeddings[
            seq_pos : seq_pos + res.shape[1], :
        ].unsqueeze(0)

        mstart = seq_pos
        mend = seq_pos + res.shape[1]
        for layer in self.layers:
            res = layer(
                res,
                mask=self.mask[mstart:mend, :mend],
                seq_pos=seq_pos,
                use_kv_cache=use_kv_cache,
            )

        result = self.output(self.norm(res))
        return result
        ### END YOUR CODE

    def load_llama_weights(self, checkpoint):
        self.embedding.weight.data = checkpoint['tok_embeddings.weight']
        self.pos_embeddings.data = checkpoint['pos_embeddings.weight']
        self.norm.weight.data = checkpoint['norm.weight']
        self.output.weight.data = checkpoint['output.weight']

        for i, layer in enumerate(self.layers):
            layer.attn.wq.weight.data = checkpoint[
                f'layers.{i}.attention.wq.weight'
            ]
            layer.attn.wk.weight.data = checkpoint[
                f'layers.{i}.attention.wk.weight'
            ]
            layer.attn.wv.weight.data = checkpoint[
                f'layers.{i}.attention.wv.weight'
            ]
            layer.attn.wp.weight.data = checkpoint[
                f'layers.{i}.attention.wo.weight'
            ]

            layer.mlp.w1.weight.data = checkpoint[
                f'layers.{i}.feed_forward.w1.weight'
            ]
            layer.mlp.w2.weight.data = checkpoint[
                f'layers.{i}.feed_forward.w2.weight'
            ]
            layer.mlp.w3.weight.data = checkpoint[
                f'layers.{i}.feed_forward.w3.weight'
            ]

            layer.norm1.weight.data = checkpoint[
                f'layers.{i}.attention_norm.weight'
            ]
            layer.norm2.weight.data = checkpoint[f'layers.{i}.ffn_norm.weight']


checkpoint = torch.load(
    os.path.join(here, 'consolidated.00.pth'), map_location=torch.device('cpu')
)
with open(os.path.join(here, 'params.json'), 'rt') as f:
    params = json.load(f)

model = Llama3Simplified(
    params['vocab_size'],
    params['dim'],
    params['n_heads'],
    params['max_seq_len'],
    params['dim'] * params['ffn_dim_multiplier'],
    params['n_layers'],
)
model.load_llama_weights(checkpoint)
model = model.float().to(DEVICE)


def generate(
    model, prompt_tokens, tokenizer, temp=0.7, max_tokens=500, verbose=True
):
    """
    Autoregressively sample tokens from a language model using its KV cache.

    Inputs:
        model : Module - language model mapping token sequences to logits
        prompt_tokens : list[int] - initial prompt tokens
        tokenizer : object - tokenizer with decode() and stop_tokens
        temp : float - sampling temperature
        max_tokens : int - maximum number of new tokens to generate
        verbose : bool - whether to print each generated token as it is sampled
    Output:
        list[int] - generated tokens, excluding the prompt tokens
    """
    ### BEGIN YOUR CODE
    out_tokens = []

    res = model(torch.tensor([prompt_tokens]), seq_pos=0, use_kv_cache=True)

    for i in range(max_tokens):
        p = torch.softmax(res[0, -1] / temp, dim=-1)
        next_token = torch.multinomial(p, 1).item()
        out_tokens.append(next_token)

        if verbose:
            print(tokenizer.decode([next_token]), end='', flush=True)

        if next_token in tokenizer.stop_tokens:
            break

        res = model(
            torch.tensor([[next_token]]),
            seq_pos=len(prompt_tokens) + len(out_tokens) - 1,
            use_kv_cache=True,
        )

    return out_tokens
    ### END YOUR CODE


if len(sys.argv) > 1:
    raw_prompt = sys.argv[1]
else:
    raw_prompt = 'How is the fibonacci function implemented in Python?'

tokenizer = Tokenizer(os.path.join(here, 'tokenizer.model'))
chat = ChatFormat(tokenizer)
msg = Message(role='user', content=raw_prompt)
prompt = chat.encode_dialog_prompt([msg])

generate(model, prompt, tokenizer)
print()
