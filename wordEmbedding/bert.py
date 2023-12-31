import torch

# Load rdrsegmenter from VnCoreNLP
#import py_vncorenlp
from vncorenlp import VnCoreNLP
from transformers import AutoModel, AutoTokenizer

rdrsegmenter = VnCoreNLP("/home/anh/workspace/CandidateEntityRanking/vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg",
                                                 max_heap_size='-Xmx500m')

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def get_word_after_segmentation():
	# TODO later
	pass

def reformat_entity_mention(s):
    segmented_token = rdrsegmenter.tokenize(s)[0]
    ret = []
    for token in segmented_token:
        if "-" in token:
            sub_tokens = token.split("-")
            for s_token in sub_tokens:
                ret += [s_token, "-"]
            ret = ret[:len(ret) - 1]
        else:
            ret += [token]
    return ret

def get_word_embedding(input_text):
    # To perform word (and sentence) segmentation  
    sentences = rdrsegmenter.tokenize(input_text)
    sentence = ""
    for sen in sentences:
        sentence += " ".join(sen)
    # print("after tokenize: {}".format(sentence))

    # Extract the last layer's features
    input_ids = torch.tensor([tokenizer.encode(sentence)])
    ret = []
    with torch.no_grad():
        features = phobert(input_ids) 
        last_layer_features = features.last_hidden_state
        #print(last_layer_features.size())
        # import pdb;pdb.set_trace()
        for i in len(input_ids[0]):
            # import pdb;pdb.set_trace()
            ret.append((str(tokenizer.decode(input_ids[0][i])), last_layer_features[0, i, :]))
    print(len(ret))
    return ret


def get_word_embedding_from_doc(entity_mention, sentences):
    def get_entity_mention_position(mention, sen):
        ret = []
        s = 0
        for p in range(len(sen)):
            print("here {}=== {}".format(sen[p][0], mention[s]))
            if sen[p][0].lower() == mention[s]:
                s += 1
                if s == len(mention):
                    ret.append(p - len(mention) + 1)
                    s = 0
                continue
            s = 0
        # print("Position: {}".format(ret))
        return ret
    
    def get_embeded_entity_mention(position, mention_size, sen):
        ret = []
        for i in position:
            temp_slice = [token[1] for token in sen[i:i + mention_size]]
            ret.append(torch.stack(temp_slice).mean(dim=0))
        # print("size of mention: {}".format(len(ret)))
        return ret

    tensor_stack = []
    normalized_entity_mention = [i.lower() for i in entity_mention]
    print("after normalize entity_mention {}".format(normalized_entity_mention))
    # print("length {}".format(len(normalized_entity_mention)))
    # print("length of sentence: {}".format(len(sentences)))
    # print(sentences)
    for input_doc in sentences:
        # print("input doc {}".format(input_doc.lower()))
        try:
            ret = get_word_embedding(input_doc)
        except:
            ret = []
        # Get embeded tensors of entity mention variant
        if len(normalized_entity_mention) > 1:
            # Get full size
            tensor_stack += get_embeded_entity_mention(get_entity_mention_position(normalized_entity_mention, ret), len(normalized_entity_mention), ret)
            pass
        # Get first token
        tensor_stack += get_embeded_entity_mention(get_entity_mention_position(normalized_entity_mention[0:1], ret), 1, ret)
    if len(tensor_stack) == 0:
        return None
    return torch.stack(tensor_stack).mean(dim=0)
