import torch.nn
from utils.utils import *
from utils.entity_generator import *

from wordEmbedding.bert import get_word_embedding, get_word_embedding_from_doc, reformat_entity_mention

from flask import Flask, Response, request, jsonify

app = Flask(__name__)

cos = torch.nn.CosineSimilarity(dim=0)


def calculate_similar(s, d):
    return cos(s, d)


def ranking_process(r_entity_mention, in_doc):  
    entity_mention = reformat_entity_mention(r_entity_mention)
    print("entity mention: {}, raw entity mention: {}".format(entity_mention, r_entity_mention))
    src_embed = get_word_embedding_from_doc(entity_mention, [in_doc])

    ranking_ret = []
    # Get candidates list
    page_lst = get_candidate(entity_mention=r_entity_mention)
    print(page_lst)

    # Find similarity
    for page in page_lst[:5]:
        print("Processing {}".format(page))
        page_content = get_first_paragraph(page)
        if page_content == "":
            continue
        sentences = search_sentence_with_keyword(page[0], page_content)
        print("{} length sentences: {}".format(page, len(sentences)))
        if len(sentences) != 0:
            candidate_embed = get_word_embedding_from_doc(entity_mention=reformat_entity_mention(page[0]), sentences=sentences)
            if candidate_embed is not None:
                ranking_ret.append((page, calculate_similar(src_embed, candidate_embed)))
    return ranking_ret

@app.route('/el', methods=['GET', 'POST'])
def handle_entity_linking():
    def find_most_relevant_entity(entity_mention, in_doc):
        ret = ranking_process(r_entity_mention=entity_mention, in_doc=in_doc)
        result = None
        for c in ret:
            if result is None or c[1] > result[1]:
                result = c
        return result
    
    data = request.json
    mention, doc = data["entity_mention"], data["in_doc"]
    # print("----{}---------{}".format(mention, doc))
    ret_entity = find_most_relevant_entity(entity_mention=mention, in_doc=doc)
    return jsonify({"ret": {"entity_name": ret_entity[0][0],
                            "url": "http://vi.wikipedia.org/?curid={}".format(ret_entity[0][1])}}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091)
