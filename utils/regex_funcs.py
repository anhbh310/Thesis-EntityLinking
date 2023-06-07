import re


def get_main_article(text):
    # TODO Find main article for reduce noisy sentences
    pass


def get_sentences(text):
    pat = re.compile(r'([A-ZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẴẶÉÈẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỚỜỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][^\.!?]*[\.!?])', re.UNICODE)
    return pat.findall(text)
