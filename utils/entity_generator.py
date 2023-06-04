from mediawiki import MediaWiki, mediawiki


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
