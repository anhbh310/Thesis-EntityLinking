import torch.nn
from utils.utils import *
from utils.entity_generator import *

from wordEmbedding.bert import get_word_embedding, get_word_embedding_from_doc

cos = torch.nn.CosineSimilarity(dim=0)


def calculate_similar(s, d):
    return cos(s, d)


def ranking_process(r_entity_mention, in_doc):
    entity_mention = reformat_entity_mention(r_entity_mention)
    print("entity mention: {}".format(entity_mention))
    src_embed = get_word_embedding_from_doc(entity_mention, [in_doc])

    ranking_ret = []
    # Get candidates list
    page_lst = get_candidate(entity_mention=r_entity_mention)
    print(page_lst)

    # Find similarity
    for page in page_lst[:5]:
        page_content = get_page_content(page)
        if page_content == "":
            continue
        sentences = search_sentence_with_keyword(r_entity_mention, page_content)
        if len(sentences) != 0:
            candidate_embed = get_word_embedding_from_doc(entity_mention=entity_mention, sentences=sentences)
            if candidate_embed is not None:
                ranking_ret.append((page, calculate_similar(src_embed, candidate_embed)))
    return ranking_ret


if __name__ == '__main__':
    # Input
    # ret = ranking_process(r_entity_mention="Hồng Đức", in_doc="Năm 1460, các đại thần phế truất Nghi Dân, Lê Tư Thành "
    #                                                           "được lập làm vua ,ông có 2 niên hiệu là Quang Thuận  "
    #                                                           "(1460 – 1469), Hồng Đức (1470 – 1497)")
    ret = ranking_process(r_entity_mention="Thuỵ Điển", in_doc="Santander ước tính trong một báo cáo năm 2015 rằng công nghệ này có thể tiết kiệm cho các ngân hàng khoảng 20 tỷ USD.Nhiều quốc gia phát triển như Thụy Điển đang xem xét một hệ thống đăng ký đất đai dựa trên Blockchain.Những nước khác như Ukraine và Georgia cũng đang xem xét các giải pháp dựa trên công nghệ này.Nó có thể được sử dụng để duy trì một bản ghi rõ ràng, đáng tin cậy về bất cứ điều gì.")
    print(ret)
    result = None
    for c in ret:
        if result is None or c[1] > result[1]:
            result = c
    print(result)
