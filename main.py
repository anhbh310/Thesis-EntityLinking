import gc
import sys

import torch.nn
from flask import Flask, Response, request, jsonify

from utils.utils import *
from utils.entity_generator import *

# from wordEmbedding.bert_fair import get_word_embedding, get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert import get_word_embedding, get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_multilingual import get_word_embedding_from_doc, reformat_entity_mention
from wordEmbedding.bert_multilingual_uncased import get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_xlm_roberta import get_word_embedding_from_doc, reformat_entity_mention
# from wordEmbedding.bert_vibert import get_word_embedding_from_doc, reformat_entity_mention


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
    print(page_lst[:5])
    if src_embed is None:
        print("Can not embed entity mention, return related entity from generation block".center(20, "-"))
        return [(page_lst[0], 1)]

    # Find similarity
    for page in page_lst[:5]:
        print("Processing {}".format(page))
        page_content = get_first_paragraph(page)
        if page_content == "":
            continue
        sentences = search_sentence_with_keyword(remove_type_from_entity_name(page[0]), page_content)
        print("{} length sentences: {}".format(page, len(sentences)))
        if len(sentences) != 0:
            candidate_embed = get_word_embedding_from_doc(entity_mention=reformat_entity_mention(remove_type_from_entity_name(page[0])), sentences=sentences)
            if candidate_embed is not None:
                ranking_ret.append((page, calculate_similar(src_embed, candidate_embed)))
    return ranking_ret

def ranking_process_with_multilingual(r_entity_mention, in_doc):  
    entity_mention = r_entity_mention
    print("entity mention: {}, raw entity mention: {}".format(entity_mention, r_entity_mention))
    src_embed = get_word_embedding_from_doc(entity_mention, [in_doc])

    ranking_ret = []
    # Get candidates list
    page_lst = get_candidate(entity_mention=r_entity_mention)
    print(page_lst[:5])
    if src_embed is None:
        print("Can not embed entity mention, return related entity from generation block".center(20, "-"))
        return [(page_lst[0], 1)]

    # Find similarity
    for page in page_lst[:5]:
        print("Processing {}".format(page))
        page_content = get_first_paragraph(page)
        if page_content == "":
            continue
        sentences = search_sentence_with_keyword(remove_type_from_entity_name(page[0]), page_content)
        print("{} length sentences: {}".format(page, len(sentences)))
        if len(sentences) != 0:
            candidate_embed = get_word_embedding_from_doc(entity_mention=remove_type_from_entity_name(page[0]), sentences=sentences)
            if candidate_embed is not None:
                ranking_ret.append((page, torch.nn.functional.cosine_similarity(src_embed, candidate_embed, dim=1)))
    return ranking_ret

@app.route('/el', methods=['GET', 'POST'])
def handle_entity_linking():
    def find_most_relevant_entity(entity_mention, in_doc):
        ret = ranking_process_with_multilingual(r_entity_mention=entity_mention, in_doc=in_doc)
        # ret = ranking_process(r_entity_mention=entity_mention, in_doc=in_doc)
        print(ret)
        result = None
        for c in ret:
            if result is None or c[1] > result[1]:
                result = c
        return result
    
    data = request.json
    mention, doc = data["entity_mention"], data["in_doc"]
    # print("----{}---------{}".format(mention, doc))
    ret_entity = find_most_relevant_entity(entity_mention=mention, in_doc=doc)
    if ret_entity is not None:
        return jsonify({"ret": {"entity_name": ret_entity[0][0],
                                "url": "https://vi.wikipedia.org/?curid={}".format(ret_entity[0][1])}}), 200
    gc.collect()
    return jsonify({"ret": {}}), 200
@app.route('/check', methods=['GET'])
def handle_check():
    return jsonify({"status": "READY"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091)
    # print(ranking_process_with_multilingual(
    #     r_entity_mention="cơ sở dữ liệu",
    #     in_doc="Blockchain mang lại sự thay đổi lớn ở các thị trường đang phát triển và cho phép nhiều quốc gia 'nhảy cóc' lên những cấp phát triển mới...Blockchain cho phép tất cả mọi người có chứng thực nhân thân, có tài sản để trông chờ lúc khó khăn mà khó ai có thể tước đi được Thông thường mọi người có một cơ sở dữ liệu trung tâm để ghi lại những thứ như giao dịch, thương mại.Trong các giao dịch thông thường, sẽ cần những người trung gian để đảm bảo mọi người và mọi thứ đang \"chơi\" theo đúng quy tắc.Blockchain loại bỏ sự cần thiết phải có những thành phần trung gian đó."
    #     ))
