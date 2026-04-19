import unittest
import torch
import torch.nn as nn

class TestEmbeddingLayer(unittest.TestCase):
    def setUp(self):
        # Initialize common parameters for testing
        self.num_embeddings = 10
        self.embedding_dim = 3
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def test_output_shape(self):
        # Test if the output shape matches (batch_size, embedding_dim)
        input_indices = torch.LongTensor([1, 2, 4, 5])
        output = self.embedding(input_indices)

        expected_shape = (4, 3)
        self.assertEqual(output.shape, expected_shape, "Output shape is incorrect.")

    def test_values_within_range(self):
        # Test if an index within range works and index out of range fails
        input_valid = torch.LongTensor([9]) # Last valid index
        try:
            self.embedding(input_valid)
        except IndexError:
            self.fail("Embedding raised IndexError unexpectedly for index 9")

        # Test index out of bounds (should raise error)
        input_invalid = torch.LongTensor([10])
        with self.assertRaises(IndexError):
            self.embedding(input_invalid)

    def test_gradient_flow(self):
        # Test if the weights are updated (trainable)
        input_indices = torch.LongTensor([1])
        output = self.embedding(input_indices)

        # Simulate a simple loss and backward pass
        loss = output.sum()
        loss.backward()

        # Check if the weight gradient is not None
        self.assertIsNotNone(self.embedding.weight.grad, "Gradients are not flowing to weights.")
        # Check if only the accessed row (index 1) has a non-zero gradient
        grad_row_1 = self.embedding.weight.grad[1]
        self.assertTrue(torch.any(grad_row_1 != 0), "Gradient for index 1 should be non-zero.")

    def test_padding_idx(self):
        # Test if padding_idx consistently returns zeros
        pad_idx = 0
        emb_with_pad = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=pad_idx)

        input_indices = torch.LongTensor([pad_idx])
        output = emb_with_pad(input_indices)

        # The output for index 0 must be all zeros
        self.assertTrue(torch.all(output == 0), "Padding index did not return all zeros.")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)