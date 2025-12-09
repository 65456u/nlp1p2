"""
Neural network models for sentiment classification.
Based on NLP1 2025 Practical 2.

Supported models:
- BOW: Bag of Words
- CBOW: Continuous Bag of Words
- DeepCBOW: Deep Continuous Bag of Words
- LSTMClassifier: LSTM-based classifier
- TreeLSTMClassifier: Tree-LSTM classifier
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

try:
    from .data import SHIFT, REDUCE
except ImportError:
    from data import SHIFT, REDUCE


class BOW(nn.Module):
    """A simple bag-of-words model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, vocab):
        super(BOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embed(inputs)
        logits = embeds.sum(1) + self.bias
        return logits


class CBOW(nn.Module):
    """Continuous bag-of-words model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int, vocab):
        super(CBOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs)
        x = x.sum(1)
        logits = self.output_layer(x)
        return logits


class DeepCBOW(nn.Module):
    """Deep continuous bag-of-words model with hidden layers"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 output_dim: int, vocab):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs)
        x = x.sum(1)
        logits = self.output_layer(x)
        return logits


class PTDeepCBOW(DeepCBOW):
    """Deep CBOW with pre-trained embeddings support"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, vocab):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab)


class MyLSTMCell(nn.Module):
    """Custom LSTM cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(MyLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input -> gate projections
        self.W_ii = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_if = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_ig = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_io = nn.Linear(input_size, hidden_size, bias=bias)
        
        # Hidden -> gate projections
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input_: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_h, prev_c = hx
        
        i = torch.sigmoid(self.W_ii(input_) + self.W_hi(prev_h))
        f = torch.sigmoid(self.W_if(input_) + self.W_hf(prev_h))
        g = torch.tanh(self.W_ig(input_) + self.W_hg(prev_h))
        o = torch.sigmoid(self.W_io(input_) + self.W_ho(prev_h))
        
        c = f * prev_c + i * g
        h = o * torch.tanh(c)
        
        return h, c
    
    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMClassifier(nn.Module):
    """LSTM-based sentence classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, vocab, dropout: float = 0.5):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
        
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        T = x.size(1)
        
        input_ = self.embed(x)
        
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)
        
        outputs = []
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)
        
        if B == 1:
            final = hx
        else:
            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.transpose(0, 1).contiguous()
            
            pad_positions = (x == 1).unsqueeze(-1)
            outputs = outputs.masked_fill_(pad_positions, 0.)
            
            mask = (x != 1)
            lengths = mask.sum(dim=1)
            
            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]
        
        logits = self.output_layer(final)
        return logits


class TreeLSTMCell(nn.Module):
    """Binary Tree LSTM cell"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(TreeLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
        self.dropout_layer = nn.Dropout(p=0.25)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, hx_l: Tuple[torch.Tensor, torch.Tensor],
                hx_r: Tuple[torch.Tensor, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_h_l, prev_c_l = hx_l
        prev_h_r, prev_c_r = hx_r
        
        children = torch.cat([prev_h_l, prev_h_r], dim=1)
        proj = self.reduce_layer(children)
        
        i, f_l, f_r, g, o = torch.chunk(proj, 5, dim=-1)
        
        i = torch.sigmoid(i)
        f_l = torch.sigmoid(f_l)
        f_r = torch.sigmoid(f_r)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c = i * g + f_l * prev_c_l + f_r * prev_c_r
        h = o * torch.tanh(c)
        
        return h, c
    
    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


def batch(states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turns a list of states into a single tensor for fast processing."""
    return torch.cat(states, 0).chunk(2, 1)


def unbatch(state: Tuple[torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
    """Turns a tensor back into a list of states."""
    return torch.split(torch.cat(state, 1), 1, 0)


class TreeLSTM(nn.Module):
    """Encodes a sentence using a TreeLSTMCell"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super(TreeLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.reduce = TreeLSTMCell(input_size, hidden_size)
        
        self.proj_x = nn.Linear(input_size, hidden_size)
        self.proj_x_gate = nn.Linear(input_size, hidden_size)
        
        self.buffers_dropout = nn.Dropout(p=0.5)
    
    def forward(self, x: torch.Tensor, transitions: np.ndarray) -> torch.Tensor:
        """
        Args:
            x: word embeddings [B, T, E] (reversed!)
            transitions: [2T-1, B]
        Returns:
            root states [B, D]
        """
        B = x.size(0)
        T = x.size(1)
        
        buffers_c = self.proj_x(x)
        buffers_h = buffers_c.tanh()
        buffers_h_gate = self.proj_x_gate(x).sigmoid()
        buffers_h = buffers_h_gate * buffers_h
        
        buffers = torch.cat([buffers_h, buffers_c], dim=-1)
        
        D = buffers.size(-1) // 2
        
        buffers = buffers.split(1, dim=0)
        buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]
        
        stacks = [[] for _ in buffers]
        
        for t_batch in transitions:
            child_l = []
            child_r = []
            
            for transition, buffer, stack in zip(t_batch, buffers, stacks):
                if transition == SHIFT:
                    stack.append(buffer.pop())
                elif transition == REDUCE:
                    assert len(stack) >= 2, "Stack too small!"
                    child_r.append(stack.pop())
                    child_l.append(stack.pop())
            
            if child_l:
                reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
                for transition, stack in zip(t_batch, stacks):
                    if transition == REDUCE:
                        stack.append(next(reduced))
        
        final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
        final = torch.cat(final, dim=0)
        
        return final


class TreeLSTMClassifier(nn.Module):
    """Tree-LSTM based sentence classifier"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, vocab, dropout: float = 0.5):
        super(TreeLSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.treelstm = TreeLSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
    
    def forward(self, x: Tuple[torch.Tensor, np.ndarray]) -> torch.Tensor:
        x, transitions = x
        emb = self.embed(x)
        root_states = self.treelstm(emb, transitions)
        logits = self.output_layer(root_states)
        return logits


class TreeLSTMWithNodeSupervision(nn.Module):
    """
    Tree-LSTM with node-level supervision.
    Can output predictions at each node in the tree.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 output_dim: int, vocab, dropout: float = 0.5):
        super(TreeLSTMWithNodeSupervision, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        
        # TreeLSTM components
        self.reduce = TreeLSTMCell(embedding_dim, hidden_dim)
        self.proj_x = nn.Linear(embedding_dim, hidden_dim)
        self.proj_x_gate = nn.Linear(embedding_dim, hidden_dim)
        
        # Output layer for each node
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
    
    def forward(self, x: Tuple[torch.Tensor, np.ndarray], 
                return_all_nodes: bool = False) -> torch.Tensor:
        """
        Args:
            x: (word_ids, transitions)
            return_all_nodes: if True, return predictions for all nodes
        Returns:
            logits for root (or all nodes if return_all_nodes=True)
        """
        x_ids, transitions = x
        emb = self.embed(x_ids)
        
        B = emb.size(0)
        T = emb.size(1)
        
        # Initialize leaf nodes
        buffers_c = self.proj_x(emb)
        buffers_h = buffers_c.tanh()
        buffers_h_gate = self.proj_x_gate(emb).sigmoid()
        buffers_h = buffers_h_gate * buffers_h
        
        buffers = torch.cat([buffers_h, buffers_c], dim=-1)
        
        buffers = buffers.split(1, dim=0)
        buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]
        
        stacks = [[] for _ in buffers]
        all_node_states = [[] for _ in range(B)]  # For collecting all node states
        
        # Process tree
        for t_batch in transitions:
            child_l = []
            child_r = []
            reduce_indices = []
            
            for idx, (transition, buffer, stack) in enumerate(zip(t_batch, buffers, stacks)):
                if transition == SHIFT:
                    state = buffer.pop()
                    stack.append(state)
                    # Leaf node state
                    h = state.chunk(2, -1)[0]
                    all_node_states[idx].append(h)
                elif transition == REDUCE:
                    assert len(stack) >= 2, "Stack too small!"
                    child_r.append(stack.pop())
                    child_l.append(stack.pop())
                    reduce_indices.append(idx)
            
            if child_l:
                reduced = list(unbatch(self.reduce(batch(child_l), batch(child_r))))
                for i, idx in enumerate(reduce_indices):
                    state = reduced[i]
                    stacks[idx].append(state)
                    # Internal node state
                    h = state.chunk(2, -1)[0]
                    all_node_states[idx].append(h)
        
        # Get root states
        final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
        final = torch.cat(final, dim=0)
        
        if return_all_nodes:
            # Return predictions for all nodes
            all_logits = []
            for node_states in all_node_states:
                if node_states:
                    states = torch.cat(node_states, dim=0)
                    logits = self.output_layer(states)
                    all_logits.append(logits)
            return all_logits
        else:
            logits = self.output_layer(final)
            return logits


def get_model(model_type: str, vocab_size: int, embedding_dim: int, 
              hidden_dim: int, output_dim: int, vocab, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: one of 'bow', 'cbow', 'deepcbow', 'lstm', 'lstm_batched', 
                   'treelstm', 'treelstm_each_node'
        vocab_size: vocabulary size
        embedding_dim: embedding dimension
        hidden_dim: hidden layer dimension
        output_dim: number of output classes
        vocab: vocabulary object
    
    Returns:
        model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'bow':
        return BOW(vocab_size, output_dim, vocab)
    elif model_type == 'cbow':
        return CBOW(vocab_size, embedding_dim, output_dim, vocab)
    elif model_type == 'deepcbow':
        return DeepCBOW(vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    elif model_type in ['lstm', 'lstm_batched']:
        return LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    elif model_type == 'treelstm':
        return TreeLSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    elif model_type == 'treelstm_each_node':
        return TreeLSTMWithNodeSupervision(vocab_size, embedding_dim, hidden_dim, output_dim, vocab)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_parameters(model: nn.Module):
    """Print model parameters."""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(
            name, str(list(p.shape)), p.requires_grad))
    print("\nTotal number of parameters: {}\n".format(total))
