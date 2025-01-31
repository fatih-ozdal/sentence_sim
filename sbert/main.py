from sentence_transformers import SentenceTransformer
import torch

def main():
    sentences = [
        "The sun is shining brightly today.",
        "It is a warm and sunny day outside.",
        "I love reading books on a rainy afternoon.",
        "The weather is perfect for a picnic.",
        "Reading books is a great way to learn new things."
    ]

    model_name = "all-mpnet-base-v2"
    model = load_model(model_name)
    sentence_vectors = encode_sentences(model, sentences)
    most_similar_pair, similarity_score = find_most_similar_pair(sentence_vectors)
    print_results(sentences, most_similar_pair, similarity_score)

def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def encode_sentences(model: SentenceTransformer, sentences: list) -> torch.tensor:
    return torch.tensor(model.encode(sentences))

def find_most_similar_pair(sentence_vectors: torch.tensor) -> tuple:
    max_similarity = 0
    most_similar_pair = (-1, -1)
    
    num_vectors = len(sentence_vectors)
    for i in range(num_vectors - 1):
        for j in range(i + 1, num_vectors):
            similarity_score = torch.nn.functional.cosine_similarity(
                sentence_vectors[i].unsqueeze(0),
                sentence_vectors[j].unsqueeze(0),
                dim=1
            ).item()

            if similarity_score > max_similarity:
                max_similarity = similarity_score
                most_similar_pair = (i, j)
    
    return most_similar_pair, max_similarity

def print_results(sentences: list, most_similar_pair: tuple, similarity_score: float) -> None :
    first_index, second_index = most_similar_pair
    print("Most similar sentences are:")
    print(f"{first_index + 1}) {sentences[first_index]}")
    print(f"{second_index + 1}) {sentences[second_index]}")
    print(f"With {100 * similarity_score:.2f}% similarity.")

if __name__ == "__main__":
    main()
