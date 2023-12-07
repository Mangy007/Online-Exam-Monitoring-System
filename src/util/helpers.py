from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity(arr1, arr2) -> int:
    '''Returns the Jaccard similarity between two lists.

    Parameters
    ----------
    arr1, arr2 : Numpy array
        Numpy arrays to compare.

    Returns
    -------
    out: Int
        Cosine similarity between the two numpy array.

    '''
    similarity_score = cosine_similarity(arr1.reshape(1,-1), arr2.reshape(1,-1))
    return similarity_score[0][0]