from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Ingest prompts
# ---------------------------------------------------------------------------

SUMMARIZE_SOURCE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor. Your job is to write a clear, well-structured "
        "markdown summary of a source document that will be stored in a personal wiki.\n\n"
        "Format:\n"
        "# <Title>\n\n"
        "**Source:** <brief origin description>\n\n"
        "## Summary\n<2-4 paragraph overview>\n\n"
        "## Key Points\n- <bullet list>\n\n"
        "## Tags\n<comma-separated topic tags>"
    )),
    ("human", "Summarize this source document:\n\n{source_content}"),
])

EXTRACT_ENTITIES_AND_CONCEPTS = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor extracting structured information from a document summary.\n\n"
        "Return a JSON object with two keys:\n"
        "- \"entities\": list of objects with keys: name (str), description (str), category (str — "
        "e.g. Person, Organization, Place, Product, Event)\n"
        "- \"concepts\": list of objects with keys: name (str), description (str)\n\n"
        "Only include entities and concepts that are genuinely important to the source. "
        "Return valid JSON only, no markdown fences."
    )),
    ("human", "Extract entities and concepts from this summary:\n\n{summary}"),
])

UPDATE_ENTITY_PAGE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor maintaining an entity page in a wiki. "
        "Update or create the page for the entity, integrating new information from a source. "
        "Preserve existing accurate content. Note contradictions explicitly. "
        "Use markdown. Include a 'Sources' section with backlinks using [[page]] syntax."
    )),
    ("human", (
        "Entity: {entity_name}\n\n"
        "Existing page content (empty if new):\n{existing_content}\n\n"
        "New information from source:\n{new_info}\n\n"
        "Source page reference: [[{source_ref}]]"
    )),
])

UPDATE_CONCEPT_PAGE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor maintaining a concept page in a wiki. "
        "Update or create the page for the concept, integrating new information. "
        "Preserve existing accurate content. Note contradictions explicitly. "
        "Use markdown. Include a 'See Also' section with [[links]] to related pages."
    )),
    ("human", (
        "Concept: {concept_name}\n\n"
        "Existing page content (empty if new):\n{existing_content}\n\n"
        "New information from source:\n{new_info}\n\n"
        "Source page reference: [[{source_ref}]]"
    )),
])

UPDATE_INDEX = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor maintaining an index file for a wiki. "
        "The index lists all pages organized by category (Sources, Entities, Concepts). "
        "Each entry format: `- [[rel/path]] — one-line description`\n\n"
        "Update the index to include the new/updated pages. "
        "Keep existing entries unless a page was replaced. "
        "Return the complete updated index.md content."
    )),
    ("human", (
        "Current index.md:\n{current_index}\n\n"
        "Pages written in this ingest (rel_path | title | one-line summary):\n{pages_summary}"
    )),
])

# ---------------------------------------------------------------------------
# Query prompts
# ---------------------------------------------------------------------------

FIND_RELEVANT_PAGES = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base search assistant. "
        "Given a question and a wiki index, identify which pages are most relevant to answer it.\n\n"
        "Return a JSON array of relative file paths (strings). "
        "Include at most 10 pages. Return valid JSON only, no markdown fences."
    )),
    ("human", "Question: {question}\n\nWiki index:\n{index_content}"),
])

SYNTHESIZE_ANSWER = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base assistant. Answer the question using only the provided wiki pages. "
        "Cite sources using [[page/path]] inline. If information is missing, say so. "
        "Format your answer in clear markdown."
    )),
    ("human", (
        "Question: {question}\n\n"
        "Relevant wiki pages:\n\n{page_contents}"
    )),
])

GENERATE_ANSWER_PAGE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base editor. Turn this Q&A into a wiki page worth keeping. "
        "Give it a clear title, summarize the question as context, present the answer clearly, "
        "and add a 'See Also' section with [[links]]. Use markdown."
    )),
    ("human", "Question: {question}\n\nAnswer:\n{answer}"),
])

# ---------------------------------------------------------------------------
# Lint prompts
# ---------------------------------------------------------------------------

LINT_WIKI = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a knowledge-base quality auditor. Analyze the wiki pages and find issues.\n\n"
        "Return a JSON array of issue objects, each with keys:\n"
        "- type: one of 'contradiction', 'stale_claim', 'orphan', 'missing_page', "
        "'missing_crossref', 'incomplete'\n"
        "- page: relative path of the affected page\n"
        "- description: what the issue is\n"
        "- suggestion: how to fix it\n\n"
        "Return valid JSON only, no markdown fences. Empty array if no issues found."
    )),
    ("human", "Wiki pages (path: content):\n\n{all_pages_text}"),
])
