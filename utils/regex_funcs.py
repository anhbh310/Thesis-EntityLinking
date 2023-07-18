import re


def get_sentences(text):
    pat = re.compile(r'([A-ZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẴẶÉÈẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỚỜỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][^\.!?]*[\.!?])', re.UNICODE)
    return pat.findall(text)

def remove_type_from_entity_name(text):
    tmp = re.sub(r'\([^()]*\)', '', text)
    if ',' in tmp:
        return tmp[:tmp.find(",")]
    return re.sub(r'\([^()]*\)', '', text)