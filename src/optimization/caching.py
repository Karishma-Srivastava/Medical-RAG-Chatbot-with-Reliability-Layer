cache = {}

def get_from_cache(query):
    return cache.get(query)

def save_to_cache(query, response):
    cache[query] = response