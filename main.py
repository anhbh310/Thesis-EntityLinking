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
    # if len(ranking_ret) == 0:
    #     print("Can not find ")
    return ranking_ret

@app.route('/el', methods=['GET', 'POST'])
def handle_entity_linking():
    def find_most_relevant_entity(entity_mention, in_doc):
        ret = ranking_process(r_entity_mention=entity_mention, in_doc=in_doc)
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
    return jsonify({"ret": {}}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8091)
    # print(ranking_process(r_entity_mention="Brazil",
    #                       in_doc="Bạn có thể nói các ngôn ngữ khác: Trợ lý ảo Google Assistant có khả năng nói các ngôn ngữ khác nhau ở các quốc gia trên thế giới.Chúng sẽ nói tiếng Ý khi ở Ý, tiếng Tây Ban Nha ở Mexico, tiếng Pháp ở Canada, tiếng Bồ Đào Nha ở Brazil...Bạn có thể kiểm tra danh sách đầy đủ các ngôn ngữ có sẵn trên blog của Google.11.Tìm thợ ống nước, người làm vườn... ở gần khu vực nhà bạn: Chỉ cần nói OK Google, find me a plumber, Google Home sẽ liệt kê những người thợ sửa ống nước gần khu vực bạn ở để bạn lựa chọn.12."))
