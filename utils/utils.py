from .regex_funcs import *


def search_sentence_with_keyword(r_entity_mention, page_content):
    # Get all line in page content
    sentences = get_sentences(page_content)
    ret = []
    for s in sentences:
        if r_entity_mention.lower() in s.lower():
            ret.append(s)
    return ret
