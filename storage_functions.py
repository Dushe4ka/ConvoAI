def filter_relevant_texts(search_results: list, max_length=512):
    """
    Отфильтровывает нерелевантные результаты поиска.
    """
    relevant_texts = []
    for res in search_results:
        text = res.payload.get("text", "")
        if text and len(text.split()) < max_length:
            relevant_texts.append(text)
    return relevant_texts