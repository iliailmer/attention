from src.utils import read_data
from src.tokenization import Tokenizer
from src.transformer import TransformerModel

text = read_data()
tokenizer = Tokenizer(from_text=text)

tokens = tokenizer.encode("Hello, World!")
print(tokens)
print(tokenizer.decode(tokens))

model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    embedding_size=8,
    head_size=8,
    block_size=8,
    num_encoders=2,
    num_decoders=2,
)

print(model)
