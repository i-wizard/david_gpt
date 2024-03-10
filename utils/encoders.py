from typing import List


class Encoder:
    """
    This is a character level tokenizer
    This means it creates an integer token for
    each character in a text sequence.
    The result is that we'll have a vocab size of about
    65 unique characters, 
    a vocab size this small means we'll have longer tokens for each word
    >>>> characters sample ['\n', ' ', '!', '$', '&', "'", ',', 
        '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D',
        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
        'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    """
    
    def __init__(self, characters: List[str]):
        self.characters = characters
        
        
    def encode(self, string: str) -> List[int]:
        characters = self.characters
        encode_dict = {character: index for index, character in enumerate(characters)}
        encoded_data: List[int] = [encode_dict[character] for character in string]
        return encoded_data

    def decode(self, encoded_data: List[str]) -> str:
        characters = self.characters
        decode_dict = {index: character for index, character in enumerate(characters)}
        decoded_data_sequence: List[str] = [decode_dict[index] for index in encoded_data]
        decoded_data: str = "".join(decoded_data_sequence)
        return decoded_data