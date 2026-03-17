from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vec1, vec2):
    score = cosine_similarity(vec1, vec2)
    return float(score[0][0])