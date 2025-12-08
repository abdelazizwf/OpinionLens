from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer_porter(word_list: list[str]) -> list[str]:
    return [porter.stem(word) for word in word_list]


def tokenizer(text: str) ->list[str]:
    return text.split()
