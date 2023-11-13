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


def sequence_to_bieso(sequence):
    """
    Converts a sequence of labels to BIESO format.
    """
    bieso_sequence = []
    for i, label in enumerate(sequence):
        if label == 'O':  # Outside any entity
            bieso_sequence.append('O')
        elif i > 0 and sequence[i-1] == label:  # Inside an entity
            if i < len(sequence) - 1 and sequence[i+1] == label:
                bieso_sequence.append('I-' + label)
            else:
                bieso_sequence.append('E-' + label)
        else:  # Beginning of an entity or single-entity
            if i < len(sequence) - 1 and sequence[i+1] == label:
                bieso_sequence.append('B-' + label)
            else:
                bieso_sequence.append('S-' + label)
    return bieso_sequence


def convert_sequence_to_tags(sequence, tagging_scheme):
    """
    Converts a sequence of labels to the specified tagging scheme.
    """
    if tagging_scheme == 'BIO':
        return sequence_to_bio(sequence)
    elif tagging_scheme == 'BIESO':
        return sequence_to_bieso(sequence)
    else:
        raise ValueError(f'Invalid tagging scheme: {tagging_scheme}.')
