__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from transformers import AutoTokenizer, AutoModelForMaskedLM

phobert_collection, multilingual_collection = None, None
persist_directory = "./persist"
client = chromadb.PersistentClient(path=persist_directory)

def get_collection(collection_name):
    global phobert_collection, multilingual_collection

    if collection_name == "phobert" and phobert_collection is not None:
        return phobert_collection
    if collection_name == "multilingual_uncased" and multilingual_collection is not None:
        return multilingual_collection
    
    # If collection is not existent
    try:
        ret = client.get_collection(name=collection_name)
    except ValueError as e:
        # print(e)
        ret = client.create_collection(name=collection_name)
    return ret

# PhoBERT embeddings
# Multilingual uncased
def add_data(collection_name, data, meta, page_id):
    collection = get_collection(collection_name)

    collection.add(
        embeddings=[data],
        metadatas=[meta],
        ids=[str(page_id)]
    )


def get_data(collection_name, page_id):
    collection = get_collection(collection_name)
    ret = collection.get(include=['embeddings'], ids=[str(page_id)])
    if len(ret["embeddings"]) ==0:
        return None
    return ret["embeddings"]


if __name__ == "__main__":
    # add_data("phobert", [1,2,3], {"entity": "entity name"}, "1")
    # add_data("phobert", [1,3,5], {"entity": "entity name"}, "2")
    # add_data("multilingual_uncased", [4,5,6,7,8], {"entity": "entity name"}, "1")
    # print("Start getting items")
    # print(get_data("phobert", 1))

    # Test with transformers
    # tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    # model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-large")

    # third_doc= "Sáng 19/5, tại đền thờ Bác Hồ ở xã An Thạnh Đông, huyện Cù Lao Dung, Đảng bộ và nhân dân Sóc Trăng long trọng tổ chức kỷ niệm 127 năm ngày sinh của Chủ tịch Hồ Chí Minh với sự tham gia của hàng ngàn người dân địa phương.Tại buổi lễ, các đại biểu đã ôn lại cuộc đời, sự nghiệp của Chủ tịch Hồ Chí Minh, vào đền thờ dâng hương, báo công với Bác.Riêng người dân địa phương tổ chức nhiều mâm cỗ giản dị, trang trọng là sản phẩm do bà con làm ra để kính dâng Bác với tất cả tấm lòng thành kính của người dân vùng cù lao bốn bề sông nước dâng Bác trong ngày sinh nhật Người."
    # third_encoded_input = tokenizer(third_doc, return_tensors='pt')
    # third_output = model.forward(third_encoded_input.input_ids, output_hidden_states=True).hidden_states[-1]
    # add_data("phobert", third_output.tolist(), {"entity": "entity name"}, "1")
    # print("done")

    # t = get_collection("phobert")
    print(get_data("phobert", 42197))
    import pdb; pdb.set_trace()
