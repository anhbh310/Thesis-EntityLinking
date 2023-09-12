from mediawiki import MediaWiki, mediawiki
import mwparserfromhell
import re


@mediawiki.memoize
def custom_search(m_wiki, query, results=10, suggestion=False):
    """ Search for similar titles

                Args:
                    m_wiki (str): MediaWiki instance
                    query (str): Page title
                    results (int): Number of pages to return
                    suggestion (bool): Use suggestion
                Returns:
                    tuple or list: tuple (list results, suggestion) if \
                                   suggestion is **True**; list of results \
                                   otherwise
                Note:
                    Could add ability to continue past the limit of 500
            """

    m_wiki._check_query(query, "Query must be specified")

    max_pull = 500

    search_params = {
        "list": "search",
        "srprop": "",
        "srlimit": min(results, max_pull) if results is not None else max_pull,
        "srsearch": query,
        "sroffset": 0,  # this is what will be used to pull more than the max
    }
    if suggestion:
        search_params["srinfo"] = "suggestion"

    raw_results = m_wiki.wiki_request(search_params)

    m_wiki._check_error_response(raw_results, query)

    search_results = [(d["title"], d["pageid"]) for d in raw_results["query"]["search"]]

    if suggestion:
        sug = None
        if raw_results["query"].get("searchinfo"):
            sug = raw_results["query"]["searchinfo"]["suggestion"]
        return search_results, sug
    return search_results


wikipedia = MediaWiki(user_agent='pyMediaWiki-User-Agent-String')
wikipedia.language = "vi"


# Search via custom function
def get_candidate(entity_mention):
    global wikipedia
    p = custom_search(wikipedia, query=entity_mention, results=50)
    # return [(c[0], c[1]) for c in p if entity_mention in c[0]]
    return [(c[0], c[1]) for c in p]


def get_page_content(page_id):
    global wikipedia
    try:
        p = wikipedia.page(pageid=page_id)
    except:
        return ""
    return p.content

def get_first_paragraph(page_id):
    def parse_section(s):
        opn = [m.start() for m in re.finditer('{{', s)]
        cls = [m.start() for m in re.finditer('}}', s)]
        # print(len(opn), len(cls))
        # print(opn)
        # print(cls)
        opn.append(1e20)
        ret = []
        d, c = 0, 0
        pivot = 0
        start = -1
        depth = 0
        if len(cls) > 0 and opn[-1] < 1500:
            while pivot < cls[-1]:
                pivot = min(opn[d], cls[c])
                if start == -1:
                    start = pivot
                if pivot == opn[d]:
                    depth += 1
                    if d < len(opn) - 1:
                        d += 1
                else:
                    depth -= 1
                    if depth == 0:
                        ret.append(s[start:cls[c]+2])
                        start = cls[c]+3
                    c += 1
                # print(d, c, depth)
        else:
            start = 0
        ret.append(s[start:])
        counter = 0
        # print(ret)
        for i in range(len(ret)):
            obj = ret[i]
            if obj[0:2] == "{{" and obj[-2:] == "}}":
                counter += 1
        ret = ret[counter:]
        first_paragraph = "".join(part for part in ret)

        return first_paragraph
    
    global wikipedia
    raw_wiki = wikipedia.wiki_request(params={"action":"parse", "pageid":page_id, "section":0, "prop":"wikitext"})
    return mwparserfromhell.parse(parse_section(str(raw_wiki["parse"]["wikitext"]["*"]))).strip_code()
    
