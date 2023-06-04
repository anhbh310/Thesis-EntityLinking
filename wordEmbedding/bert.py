import torch

# Load rdrsegmenter from VnCoreNLP
#import py_vncorenlp
from vncorenlp import VnCoreNLP
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options

rdrsegmenter = VnCoreNLP("/home/anh/workspace/CandidateEntityRanking/vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg",
                                                 max_heap_size='-Xmx500m')
#py_vncorenlp.download_model(save_dir='/home/anh/workspace/CandidateEntityRanking/vncorenlp')
#rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/home/anh/workspace/CandidateEntityRanking/vncorenlp')

phobert = RobertaModel.from_pretrained("/home/anh/workspace/CandidateEntityRanking/PhoBERT_base_fairseq", checkpoint_file='model.pt')

# Incorporate the BPE encoder into PhoBERT-base
parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
					default="/home/anh/workspace/CandidateEntityRanking/PhoBERT_base_fairseq/bpe.codes")
args = parser.parse_args()
# Incorporate the BPE encoder into PhoBERT
phobert.bpe = fastBPE(args)


def get_word_after_segmentation():
	# TODO later
	pass


def get_word_embedding(input_text):
	# To perform word (and sentence) segmentation
	#sentences = rdrsegmenter.word_segment(input_text)
        sentences = rdrsegmenter.tokenize(input_text)
	# print(sentences)
        sentence = ""
        for sen in sentences:
            sentence += " ".join(sen)
	# print(sentence)
	
	# Extract the last layer's features
        last_layer_features = phobert.extract_features_aligned_to_words(sentence)
        #print(last_layer_features.size())
        ret = []
        for tok in last_layer_features:
		# print('{:10}{} (...) {}'.format(str(tok), tok.vector[:5], tok.vector.size()))
            ret.append((str(tok), tok.vector))
        return ret


def get_word_embedding_from_doc(entity_mention, sentences):
	tensor_stack = []
	for input_doc in sentences:
            try:
                ret = get_word_embedding(input_doc)
            except:
                ret = []
            for t in ret:
                if t[0] == entity_mention:
                    tensor_stack.append(t[1])
            if len(tensor_stack) == 0:
                    return None
	return torch.stack(tensor_stack).mean(dim=0)
