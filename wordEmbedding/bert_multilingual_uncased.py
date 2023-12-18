from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

import sys
sys.path.insert(0, '..')
from utils.chroma_cli import add_data, get_data
from utils.utils import *
from utils.entity_generator import *

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-uncased")
collection_name = "multilingual_uncased"

def reformat_entity_mention(s):
    return s

def find_mention_similarity(embedding_first, embedding_second):
    return torch.nn.functional.cosine_similarity(embedding_first, embedding_second, dim=2)

def get_embedding(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output.logits, encoded_input.input_ids

def get_entity_mention_position(encoded_doc, encoded_mention):
    ret = []
    for i in range(0, len(encoded_doc)):
        if torch.equal(encoded_doc[i:i+len(encoded_mention)-2], encoded_mention[1: len(encoded_mention)-1]):
            ret.append(i)
    return ret

def get_word_embedding_from_doc(entity_mention, sentences, page_id=None):

    def embed():
        # Embed the entity_mention
        embeded_mention, tokenized_mention = get_embedding(entity_mention)
        ret = []

        for sen in sentences:
            embedding, tokenized_sen = get_embedding(sen)

            matching_pos = get_entity_mention_position(tokenized_sen[0, :], tokenized_mention[0, :])

            # Get all the embeded_candidate
            for p in matching_pos:
                ret.append(embedding[:, p:p+len(tokenized_mention[0, :])-2, :])
        # import pdb; pdb.set_trace()
        if len(ret) == 0:
            return None
        return torch.stack(ret).mean(dim=0).mean(dim=1).detach()
    
    if page_id is not None:
        # Query from chromadb
        d = get_data(collection_name, page_id)
        if d is not None:
            return torch.tensor(d)
        
        # Get first paragraph
        page_content = get_first_paragraph(page_id)
        if page_content == "":
            print("Can not get the first paragraph")
            return None

        # Get sentences that include the mention
        sentences = search_sentence_with_keyword(remove_type_from_entity_name(entity_mention), page_content)
        # print("mention: {}".format(remove_type_from_entity_name(entity_mention)))
        # print("content: {}".format(page_content))
        if len(sentences) == 0:
            print("Can not get included mention sentences")
            return None

        embed_ret = embed()
        add_data(collection_name, embed_ret[0].tolist(), {"entity": entity_mention}, page_id)
        return embed_ret
    else:
        embed_ret = embed()
        return embed_ret
        
