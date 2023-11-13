def sequence_to_bio(sequence):
    """
    Converts a sequence of labels to BIO format.
    """
    bio_sequence = []
    for i, label in enumerate(sequence):
        if i == 0:
            if label != 'O':
                bio_sequence.append('B-' + label)
            else:
                bio_sequence.append(label)
        else:
            if label != 'O':
                if sequence[i - 1] != label:
                    bio_sequence.append('B-' + label)
                else:
                    bio_sequence.append('I-' + label)
            else:
                bio_sequence.append(label)
    return bio_sequence


def sequence_to_bioes(sequence):
    """
    Converts a sequence of labels to BIESO format.
    """
    bioes_sequence = []
    for i, label in enumerate(sequence):
        if label == 'O':  # Outside any entity
            bioes_sequence.append('O')
        elif i > 0 and sequence[i-1] == label:  # Inside an entity
            if i < len(sequence) - 1 and sequence[i+1] == label:
                bioes_sequence.append('I-' + label)
            else:
                bioes_sequence.append('E-' + label)
        else:  # Beginning of an entity or single-entity
            if i < len(sequence) - 1 and sequence[i+1] == label:
                bioes_sequence.append('B-' + label)
            else:
                bioes_sequence.append('S-' + label)
    return bioes_sequence

def convert_sequence_to_tags(sequence, tagging_scheme):
    """
    Converts a sequence of labels to the specified tagging scheme.
    """
    if tagging_scheme == 'BIO':
        return sequence_to_bio(sequence)
    elif tagging_scheme == 'BIOES':
        return sequence_to_bioes(sequence)
    else:
        raise ValueError(f'Invalid tagging scheme: {tagging_scheme}.')

