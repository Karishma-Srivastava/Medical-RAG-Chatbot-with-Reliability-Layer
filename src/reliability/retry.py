def should_retry(confidence, query):

    length = len(query.split())

    # very short query → don't retry
    if length < 3:
        return False

    # adaptive threshold
    if length < 5:
        threshold = 0.7
    elif length < 12:
        threshold = 0.6
    else:
        threshold = 0.5

    return confidence < threshold