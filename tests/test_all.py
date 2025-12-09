"""
Unit tests for the sentiment analysis experiment framework.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    Vocabulary, Example, tokens_from_treestring, transitions_from_treestring,
    pad, extract_all_subtrees, build_augmented_dataset,
    SHIFT, REDUCE, NUM_CLASSES
)
from models import (
    BOW, CBOW, DeepCBOW, LSTMClassifier, TreeLSTMClassifier,
    TreeLSTMWithNodeSupervision, MyLSTMCell, TreeLSTMCell,
    get_model
)
from train import (
    ExperimentConfig, ExperimentMetrics, ExperimentResult,
    set_seed, get_device, prepare_example, prepare_minibatch,
    prepare_treelstm_minibatch
)


class TestVocabulary(unittest.TestCase):
    """Tests for Vocabulary class."""
    
    def test_vocabulary_creation(self):
        """Test basic vocabulary creation."""
        v = Vocabulary()
        v.add_token("<unk>")
        v.add_token("<pad>")
        v.add_token("hello")
        v.add_token("world")
        
        self.assertEqual(len(v), 4)
        self.assertEqual(v.w2i["<unk>"], 0)
        self.assertEqual(v.w2i["<pad>"], 1)
        self.assertEqual(v.w2i["hello"], 2)
        self.assertEqual(v.i2w[2], "hello")
    
    def test_vocabulary_build(self):
        """Test vocabulary building with frequency cutoff."""
        v = Vocabulary()
        for word in ["the", "the", "a", "a", "cat", "dog", "rare"]:
            v.count_token(word)
        
        v.build(min_freq=2)
        
        # <unk>, <pad>, the, a should be in vocab
        self.assertIn("<unk>", v.w2i)
        self.assertIn("<pad>", v.w2i)
        self.assertIn("the", v.w2i)
        self.assertIn("a", v.w2i)
        # rare words should not be in vocab
        self.assertNotIn("rare", v.w2i)


class TestDataProcessing(unittest.TestCase):
    """Tests for data processing functions."""
    
    def test_tokens_from_treestring(self):
        """Test token extraction from tree string."""
        s = "(3 (2 It) (4 (2 's) (4 good)))"
        tokens = tokens_from_treestring(s)
        self.assertEqual(tokens, ["It", "'s", "good"])
    
    def test_transitions_from_treestring(self):
        """Test transition extraction from tree string."""
        s = "(3 (2 It) (4 (2 's) (4 good)))"
        trans = transitions_from_treestring(s)
        
        # Should have SHIFTs and REDUCEs
        self.assertTrue(all(t in [SHIFT, REDUCE] for t in trans))
        # Number of SHIFTs should equal number of words
        self.assertEqual(trans.count(SHIFT), 3)
    
    def test_pad(self):
        """Test padding function."""
        tokens = [1, 2, 3]
        padded = pad(tokens, 5, pad_value=0)
        self.assertEqual(padded, [1, 2, 3, 0, 0])
        
        # Padding shorter
        padded = pad(tokens, 3)
        self.assertEqual(padded, [1, 2, 3])


class TestModels(unittest.TestCase):
    """Tests for model architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 100
        self.embedding_dim = 50
        self.hidden_dim = 32
        self.output_dim = NUM_CLASSES
        
        # Create a simple vocabulary mock
        class MockVocab:
            w2i = {f"word{i}": i for i in range(100)}
            i2w = [f"word{i}" for i in range(100)]
        
        self.vocab = MockVocab()
        self.device = torch.device('cpu')
    
    def test_bow_model(self):
        """Test BOW model forward pass."""
        model = BOW(self.vocab_size, self.output_dim, self.vocab)
        x = torch.LongTensor([[1, 2, 3, 4]])
        output = model(x)
        
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_cbow_model(self):
        """Test CBOW model forward pass."""
        model = CBOW(self.vocab_size, self.embedding_dim, self.output_dim, self.vocab)
        x = torch.LongTensor([[1, 2, 3, 4]])
        output = model(x)
        
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_deepcbow_model(self):
        """Test DeepCBOW model forward pass."""
        model = DeepCBOW(self.vocab_size, self.embedding_dim, self.hidden_dim, 
                         self.output_dim, self.vocab)
        x = torch.LongTensor([[1, 2, 3, 4]])
        output = model(x)
        
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_lstm_model(self):
        """Test LSTM model forward pass."""
        model = LSTMClassifier(self.vocab_size, self.embedding_dim, self.hidden_dim,
                               self.output_dim, self.vocab)
        x = torch.LongTensor([[1, 2, 3, 4]])
        output = model(x)
        
        self.assertEqual(output.shape, (1, self.output_dim))
    
    def test_lstm_batched(self):
        """Test LSTM model with batched input."""
        model = LSTMClassifier(self.vocab_size, self.embedding_dim, self.hidden_dim,
                               self.output_dim, self.vocab)
        # Batch of 3 sentences with padding
        x = torch.LongTensor([
            [1, 2, 3, 1, 1],  # padded with 1
            [1, 2, 3, 4, 5],
            [1, 2, 1, 1, 1],  # short sentence
        ])
        output = model(x)
        
        self.assertEqual(output.shape, (3, self.output_dim))
    
    def test_lstm_cell(self):
        """Test custom LSTM cell."""
        cell = MyLSTMCell(self.embedding_dim, self.hidden_dim)
        
        x = torch.randn(1, self.embedding_dim)
        hx = torch.zeros(1, self.hidden_dim)
        cx = torch.zeros(1, self.hidden_dim)
        
        h, c = cell(x, (hx, cx))
        
        self.assertEqual(h.shape, (1, self.hidden_dim))
        self.assertEqual(c.shape, (1, self.hidden_dim))
    
    def test_treelstm_cell(self):
        """Test TreeLSTM cell."""
        cell = TreeLSTMCell(self.embedding_dim, self.hidden_dim)
        
        h_l = torch.randn(1, self.hidden_dim)
        c_l = torch.randn(1, self.hidden_dim)
        h_r = torch.randn(1, self.hidden_dim)
        c_r = torch.randn(1, self.hidden_dim)
        
        h, c = cell((h_l, c_l), (h_r, c_r))
        
        self.assertEqual(h.shape, (1, self.hidden_dim))
        self.assertEqual(c.shape, (1, self.hidden_dim))
    
    def test_get_model_factory(self):
        """Test model factory function."""
        for model_type in ['bow', 'cbow', 'deepcbow', 'lstm', 'lstm_batched']:
            model = get_model(model_type, self.vocab_size, self.embedding_dim,
                            self.hidden_dim, self.output_dim, self.vocab)
            self.assertIsNotNone(model)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            get_model('invalid_model', self.vocab_size, self.embedding_dim,
                     self.hidden_dim, self.output_dim, self.vocab)


class TestTraining(unittest.TestCase):
    """Tests for training utilities."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        r1 = torch.rand(5)
        
        set_seed(42)
        r2 = torch.rand(5)
        
        self.assertTrue(torch.allclose(r1, r2))
    
    def test_experiment_config(self):
        """Test experiment configuration."""
        config = ExperimentConfig(
            model_type='lstm',
            num_iters=1000,
            learning_rate=0.001,
            batch_size=32
        )
        
        d = config.to_dict()
        self.assertEqual(d['model_type'], 'lstm')
        self.assertEqual(d['num_iters'], 1000)
    
    def test_experiment_metrics(self):
        """Test experiment metrics."""
        metrics = ExperimentMetrics()
        metrics.train_losses = [1.0, 0.8, 0.6]
        metrics.dev_accuracies = [0.3, 0.5, 0.6]
        metrics.test_accuracy = 0.55
        
        d = metrics.to_dict()
        self.assertEqual(len(d['train_losses']), 3)
        self.assertEqual(d['test_accuracy'], 0.55)
    
    def test_experiment_result_save_load(self):
        """Test saving and loading experiment results."""
        config = ExperimentConfig(
            model_type='lstm',
            num_iters=1000,
            learning_rate=0.001,
            batch_size=32
        )
        metrics = ExperimentMetrics()
        metrics.test_accuracy = 0.55
        
        result = ExperimentResult(config=config, metrics=metrics)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            result.save(path)
            loaded = ExperimentResult.load(path)
            
            self.assertEqual(loaded.config.model_type, 'lstm')
            self.assertEqual(loaded.metrics.test_accuracy, 0.55)
        finally:
            os.unlink(path)


class TestAugmentation(unittest.TestCase):
    """Tests for data augmentation."""
    
    def test_extract_subtrees(self):
        """Test subtree extraction."""
        from nltk import Tree
        
        tree = Tree.fromstring("(3 (2 good) (4 movie))")
        subtrees = extract_all_subtrees(tree)
        
        # Should have root + 2 children
        self.assertGreaterEqual(len(subtrees), 1)
    
    def test_build_augmented_dataset(self):
        """Test augmented dataset building."""
        from nltk import Tree
        
        # Create a simple example
        tree = Tree.fromstring("(3 (2 good) (4 movie))")
        example = Example(
            tokens=["good", "movie"],
            tree=tree,
            label=3,
            transitions=[SHIFT, SHIFT, REDUCE]
        )
        
        augmented = build_augmented_dataset([example])
        
        # Should have more examples than input
        self.assertGreater(len(augmented), 1)


class TestPrepareMinibatch(unittest.TestCase):
    """Tests for minibatch preparation."""
    
    def setUp(self):
        """Set up test fixtures."""
        class MockVocab:
            w2i = {"<unk>": 0, "<pad>": 1, "good": 2, "movie": 3, "bad": 4}
        
        self.vocab = MockVocab()
        self.device = torch.device('cpu')
        
        from nltk import Tree
        self.examples = [
            Example(
                tokens=["good", "movie"],
                tree=Tree.fromstring("(3 (2 good) (4 movie))"),
                label=3,
                transitions=[SHIFT, SHIFT, REDUCE]
            ),
            Example(
                tokens=["bad"],
                tree=Tree.fromstring("(1 bad)"),
                label=1,
                transitions=[SHIFT]
            )
        ]
    
    def test_prepare_minibatch(self):
        """Test standard minibatch preparation."""
        x, y = prepare_minibatch(self.examples, self.vocab, self.device)
        
        self.assertEqual(x.shape[0], 2)  # batch size
        self.assertEqual(y.shape[0], 2)
    
    def test_prepare_minibatch_padding(self):
        """Test that shorter sentences are padded."""
        x, y = prepare_minibatch(self.examples, self.vocab, self.device)
        
        # Second example should have padding (token ID 1)
        self.assertEqual(x[1, 1].item(), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_pipeline_bow(self):
        """Test full training pipeline with BOW model."""
        from data import load_data, build_vocabulary
        from train import run_experiment, ExperimentConfig
        
        # Skip if data not available; try common locations
        base_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(base_dir, '..', 'trees'),
            os.path.join(base_dir, '..', 'data', 'trees')
        ]
        data_dir = next((c for c in candidates if os.path.exists(c)), None)
        if data_dir is None:
            self.skipTest("Data directory not found")
        
        # Load a small amount of data
        train_data, dev_data, test_data = load_data(data_dir)
        train_data = train_data[:50]
        dev_data = dev_data[:20]
        test_data = test_data[:20]
        
        vocab = build_vocabulary(train_data)
        
        config = ExperimentConfig(
            model_type='bow',
            num_iters=100,
            learning_rate=0.01,
            batch_size=10,
            seed=42
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_experiment(
                config=config,
                train_data=train_data,
                dev_data=dev_data,
                test_data=test_data,
                vocab=vocab,
                vectors=None,
                output_dir=tmpdir
            )
            
            # Check that training produced some results
            self.assertGreater(len(result.metrics.train_losses), 0)
            self.assertGreater(result.metrics.test_accuracy, 0)


if __name__ == '__main__':
    unittest.main()
