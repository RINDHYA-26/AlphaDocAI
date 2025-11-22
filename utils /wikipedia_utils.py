import wikipedia

def wiki_search(query, sentences=4):
    try:
        return wikipedia.summary(query, sentences=sentences)
    except:
        return None
