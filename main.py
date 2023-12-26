import gc
import sys
import ast
import time

import torch.nn
from flask import Flask, Response, request, jsonify

from utils.utils import *
from utils.entity_generator import *

from wordEmbedding.bert_fair import get_word_embedding_from_doc as phobert_get_word_embedding_from_doc
from wordEmbedding.bert_fair import reformat_entity_mention as phobert_reformat_entity_mention

from wordEmbedding.bert_multilingual_uncased import get_word_embedding_from_doc

# from wordEmbedding.bert import get_word_embedding, get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_multilingual import get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_xlm_roberta import get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_vibert import get_word_embedding_from_doc, reformat_entity_mention


app = Flask(__name__)

cos = torch.nn.CosineSimilarity(dim=0)
prev_time = time.time()
is_unlinkable_part_exist = True


def measure_time():
    n = time.time()
    global prev_time
    elapsed_time = n - prev_time
    prev_time = n
    return elapsed_time

def calculate_similar(s, d):
    return cos(s, d)


def ranking_process_with_phobert(r_entity_mention, in_doc, candidate_entities):  
    entity_mention = phobert_reformat_entity_mention(r_entity_mention)
    print("entity mention: {}, raw entity mention: {}".format(entity_mention, r_entity_mention))
    src_embed = phobert_get_word_embedding_from_doc(entity_mention, [in_doc])
    print("here entity mention: {}".format(entity_mention))
    ranking_ret = []

    # print(page_lst[:5])
    if src_embed is None:
        print("Can not embed entity mention, return related entity from generation block".center(20, "-"))
        return [(candidate_entities[0], 1)]

    # Find similarity
    for page in candidate_entities[:5]:
        candidate_embed = phobert_get_word_embedding_from_doc(entity_mention=remove_type_from_entity_name(page[0]), sentences=[], page_id=page[1])
        if candidate_embed is not None:
            ranking_ret.append((page, calculate_similar(src_embed, candidate_embed)))
        else:
            ranking_ret.append((page, 0))
    return ranking_ret

def ranking_process_with_multilingual(r_entity_mention, in_doc, candidate_entities):  
    entity_mention = r_entity_mention
    print("entity mention: {}, raw entity mention: {}".format(entity_mention, r_entity_mention))
    # print("Time elapsed before embedding the input {}".format(measure_time()))
    src_embed = get_word_embedding_from_doc(entity_mention, [in_doc])
    # print("Time elapsed after embedding the input {}".format(measure_time()))

    ranking_ret = []
    
    if src_embed is None:
        print("Can not embed entity mention, return related entity from generation block".center(20, "-"))
        return [(candidate_entities[0], 1)]

    # Find similarity
    
    for page in candidate_entities[:5]:
        # print("Processing {}".format(page))
        # print("{} length sentences: {}".format(page, len(sentences)))
        # print("Time elapsed before querying the chromadb {}".format(measure_time()))
        candidate_embed = get_word_embedding_from_doc(entity_mention=remove_type_from_entity_name(page[0]), sentences=[], page_id=page[1])
        # print("Time elapsed after querying the chromadb {}".format(measure_time()))
        if candidate_embed is not None:
            ranking_ret.append((page, torch.nn.functional.cosine_similarity(src_embed, candidate_embed, dim=1)))
        else:
            ranking_ret.append((page, 0))
    
    return ranking_ret

@app.route('/el', methods=['GET', 'POST'])
def handle_entity_linking():
    def is_the_mention_unlinkable(best_candidate, ranking_list):
        for c in ranking_list:
            print(c)
            if c[0][0] == best_candidate[0] and c[1] <= 0.2:
                return True
        return False
    
    def find_most_relevant_entity(entity_mention, in_doc):
        # PhoBERT
        phobert_ranking = ranking_process_with_phobert(r_entity_mention=entity_mention, in_doc=in_doc, candidate_entities=page_lst)
        phobert_ranking.sort(key=lambda x:x[1], reverse=True)
        print("PhoBERT: {}".format(phobert_ranking))

        # Multilingual bert
        multilingual_ranking = ranking_process_with_multilingual(r_entity_mention=entity_mention, in_doc=in_doc, candidate_entities=page_lst)
        multilingual_ranking.sort(key=lambda x:x[1], reverse=True)
        print("Multilingual: {}".format(multilingual_ranking))

        # API ranking
        api_ranking = page_lst
        print("API ranking: {}".format(api_ranking))

        # Majority voting
        voting_weigh = [
            [0.4, 0.35, 0.3, 0.25, 0.2, 0.15], # Multilingual weight
            [0.5, 0.45, 0.4, 0.35, 0.3, 0.25], # PhoBERT weight
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1], # API weight
        ]

        # Votting result initialization
        p = {}
        for page, w_i in zip(page_lst[:5], list(range(5))):
            p[page] = voting_weigh[2][w_i]

        for e, w_i in zip(phobert_ranking, list(range(5))):
            p[e[0]] += voting_weigh[1][w_i]

        for e, w_i in zip(multilingual_ranking, list(range(5))):
            p[e[0]] += voting_weigh[0][w_i]
        print(p)

        max_entity = max(p, key=p.get)
        result = ast.literal_eval(str(max_entity))
        print("Time elapsed in candidate ranking {}".format(measure_time()))
        if is_unlinkable_part_exist and is_the_mention_unlinkable(result, phobert_ranking):
            return None
        print("Time elapsed in prediction unlinkable {}".format(measure_time()))
        return result
    
    print("Time elapsed before receiving the request {}".format(measure_time()))
    data = request.json
    mention, doc = data["entity_mention"], data["in_doc"]

    # Get candidates list
    page_lst = get_candidate(entity_mention=mention)
    # print("Candidates list: {}".format(page_lst))
    if len(page_lst) == 0:
        return jsonify({"ret":{"entity_name": "NIL",
                           "url": "NIL"}}), 200
    print("Time elapsed in Candidate generation duration {}".format(measure_time()))
    
    ret_entity = find_most_relevant_entity(entity_mention=mention, in_doc=doc)
    if ret_entity is not None:
        return jsonify({"ret": {"entity_name": ret_entity[0],
                                "url": "https://vi.wikipedia.org/?curid={}".format(ret_entity[1])}}), 200
    gc.collect()
    return jsonify({"ret":{"entity_name": "NIL",
                           "url": "NIL"}}), 200
@app.route('/check', methods=['GET'])
def handle_check():
    return jsonify({"status": "READY"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091)
    # print(ranking_process_with_multilingual(
    #     r_entity_mention="cơ sở dữ liệu",
    #     in_doc="Blockchain mang lại sự thay đổi lớn ở các thị trường đang phát triển và cho phép nhiều quốc gia 'nhảy cóc' lên những cấp phát triển mới...Blockchain cho phép tất cả mọi người có chứng thực nhân thân, có tài sản để trông chờ lúc khó khăn mà khó ai có thể tước đi được Thông thường mọi người có một cơ sở dữ liệu trung tâm để ghi lại những thứ như giao dịch, thương mại.Trong các giao dịch thông thường, sẽ cần những người trung gian để đảm bảo mọi người và mọi thứ đang \"chơi\" theo đúng quy tắc.Blockchain loại bỏ sự cần thiết phải có những thành phần trung gian đó."
    #     ))
