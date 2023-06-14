import re


def get_sentences(text):
    pat = re.compile(r'([A-ZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẴẶÉÈẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỚỜỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][^\.!?]*[\.!?])', re.UNICODE)
    return pat.findall(text)

def remove_type_from_entity_name(text):
    return re.sub(r'\([^()]*\)', '', text)