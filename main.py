import torch.nn
from utils.utils import *
from utils.entity_generator import *

from wordEmbedding.bert import get_word_embedding, get_word_embedding_from_doc, reformat_entity_mention

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


if __name__ == '__main__':
    # Input
    # ret = ranking_process(r_entity_mention="Cẩm Phả", in_doc="Bridgestone: Đưa trách nhiệm cộng đồng vào văn hóa doanh nghiệp.Trong không khí tất bật trước Tết, Bridgestone vẫn tâm huyết với hoạt động chăm sóc lốp và xe để khách hàng tại Cẩm Phả di chuyển an toàn qua chương trình 'Bridgestone Lăn bánh an toàn'.Diễn ra trong ba ngày từ 19 - 21/1/2018, hơn 100 ô tô đã được kiểm tra và chăm sóc tại B-select Hải Anh, 36 Lê Thanh Nghị, Cẩm Bình, Cẩm Phả.")
    ret = ranking_process(r_entity_mention="Hồ Chí Minh", in_doc="Thành phố Hồ Chí Minh hay Sài Gòn, là thành phố lớn nhất Việt Nam và là một siêu đô thị trong tương lai gần. Đây còn là trung tâm kinh tế, văn hóa, giải trí và giáo dục tại Việt Nam. Thành phố Hồ Chí Minh là thành phố trực thuộc trung ương thuộc loại đô thị đặc biệt của Việt Nam.")
    print(ret)
    result = None
    for c in ret:
        if result is None or c[1] > result[1]:
            result = c
    print(result)
