import torch
import string

DEFAULT_CHARS = list(string.ascii_lowercase)

class CharLevelTokenizer:
    """
    A character-level tokenizer that converts text into sequences of character indices
    and supports decoding indices back to text using PyTorch tensors.

    Args:
        vocab (list): A list containing all characters in the vocabulary.
        null_token (str): Token to assing for non-speech frames. Defaults to '\<NULL>'.
    """

    def __init__(self, vocab: list = DEFAULT_CHARS, null_token: str = '<NULL>'):
        self.vocab = [null_token] + vocab
        self.char_to_index = {char: idx for idx, char in enumerate(self.vocab)}
        self.index_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, chars: list) -> torch.Tensor:
        """
        Encodes a list of characters into a PyTorch tensor of character indices.

        Args:
            chars (list): A list of characters to encode.

        Returns:
            torch.Tensor: A tensor of indices representing the input characters.
        """
        try:
            indices = [self.char_to_index[char] for char in chars]
        except KeyError:
            raise Exception('Invalid input token')
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decodes a PyTorch tensor of character indices back into a string.

        Args:
            indices (torch.Tensor): A tensor of character indices.

        Returns:
            str: The decoded string.
        """
        return ''.join(self.index_to_char[idx.item()] for idx in indices if idx.item() in self.index_to_char)

    def encode_batch(self, char_lists: list) -> torch.Tensor:
        """
        Encodes a batch of lists of characters into a PyTorch tensor of indices.

        Args:
            char_lists (list): A list of lists of characters to encode.

        Returns:
            torch.Tensor: A 2D tensor of indices representing the input characters.
        """
        batch_indices = [[self.char_to_index[char] for char in chars if char in self.char_to_index] for chars in char_lists]
        return torch.tensor(batch_indices, dtype=torch.long)

    def decode_batch(self, batch_indices: torch.Tensor) -> list:
        """
        Decodes a batch of PyTorch tensors of character indices back into strings.

        Args:
            batch_indices (torch.Tensor): A 2D tensor of character indices.

        Returns:
            list: A list of decoded strings.
        """
        return [''.join(self.index_to_char[idx.item()] for idx in indices if idx.item() in self.index_to_char) 
                for indices in batch_indices]

    def add_special_tokens(self, special_tokens: list):
        """
        Adds special tokens to the vocabulary.

        Args:
            special_tokens (list): A list of special tokens to add.
        """
        for token in special_tokens:
            if token not in self.char_to_index:
                index = len(self.char_to_index)
                self.char_to_index[token] = index
                self.index_to_char[index] = token

    def vocab_size(self) -> int:
        """
        Returns the size of the tokenizer vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return len(self.char_to_index)