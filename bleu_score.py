import sacrebleu


def compute_sentence_bleu(reference, prediction):
    """
    Computes BLEU score for a single sentence.
    """
    bleu = sacrebleu.sentence_bleu(prediction, [reference])
    return bleu.score


def compute_corpus_bleu(references, predictions):
    """
    Computes corpus-level BLEU score.
    references: list of reference sentences
    predictions: list of predicted sentences
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


if __name__ == "__main__":

    print("\n🔵 BLEU Score Evaluation Tool\n")

    print("Enter Reference (Ground Truth) Sentence:")
    reference = input("Reference: ")

    print("\nEnter Model Prediction:")
    prediction = input("Prediction: ")

    score = compute_sentence_bleu(reference, prediction)

    print("\n==============================")
    print(f"Sentence BLEU Score: {score:.2f}")
    print("==============================\n")
 