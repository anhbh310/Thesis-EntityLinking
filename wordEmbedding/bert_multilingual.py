from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")

def find_mention_similarity(embedding_first, embedding_second):
    return torch.nn.functional.cosine_similarity(embedding_first, embedding_second, dim=2)

def get_embedding(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)
    return output.last_hidden_state, encoded_input.input_ids

def get_entity_mention_position(encoded_doc, encoded_mention):
    ret = []
    for i in range(0, len(encoded_doc)):
        if torch.equal(encoded_doc[i:i+len(encoded_mention)-2], encoded_mention[1: len(encoded_mention)-1]):
            ret.append(i)
    return ret

def get_word_embedding_from_doc(entity_mention, sentences):

    # Embed the entity_mention
    embeded_mention, tokenized_mention = get_embedding(entity_mention)
    similarity_score = []

    for sen in sentences:
        embedding, tokenized_sen = get_embedding(sen)

        matching_pos = get_entity_mention_position(tokenized_sen[0, :], tokenized_mention[0, :])

        # Get all the embeded_candidate
        for p in matching_pos:

            similarity_score.append(find_mention_similarity(embedding[:, p:p+len(tokenized_mention[0, :])-2, :], embeded_mention[:, 1:len(tokenized_mention[0, :])-1, :]))
    # import pdb; pdb.set_trace()
    if len(similarity_score) == 0:
        return None
    return torch.mean(torch.stack(similarity_score))
