"""
Prompt templates for the RAG system.
Contains all prompt templates used for routing, grading, and generation.
"""
from langchain_core.prompts import PromptTemplate


# Question Router: Decides whether to use vectorstore or web search
QUESTION_ROUTER_TEMPLATE = """You are an expert at routing user questions to the appropriate data source.

Use the vectorstore for questions about:
- Change management models (Lewin, Kotter, McKinsey 7S, ADKAR)
- Organizational change strategies
- Change resistance and adoption
- Transformation methodologies

Use web-search for:
- Current events
- Recent developments
- General knowledge not specific to change management

Return ONLY valid JSON using double quotes (not single quotes) with a single key "datasource" and value either "web_search" or "vectorstore".
Example: {{"datasource": "vectorstore"}}
Do not include any preamble, explanation, or markdown formatting.

Question to route: {question}"""

question_router_prompt = PromptTemplate(
    template=QUESTION_ROUTER_TEMPLATE,
    input_variables=["question"],
)


# Document Relevance Grader: Checks if retrieved documents are relevant
DOCUMENT_GRADER_TEMPLATE = """You are an expert at determining document relevance.

Given the question, evaluate if the document contains keywords or information relevant to answering it.

Return ONLY valid JSON using double quotes (not single quotes) with a single key "score" and value either "yes" or "no".
Example: {{"score": "yes"}}
Do not include any preamble, explanation, or markdown formatting.

Question: {question}
Document: {document}"""

document_grader_prompt = PromptTemplate(
    template=DOCUMENT_GRADER_TEMPLATE,
    input_variables=["question", "document"],
)


# Hallucination Grader: Checks if answer is grounded in retrieved facts
HALLUCINATION_GRADER_TEMPLATE = """You are a grader assessing whether an answer is grounded in factual information.

Here are the facts:
-------
{documents}
-------

Here is the answer: {generation}

Determine if the answer is supported by the provided facts.
Return ONLY valid JSON using double quotes (not single quotes) with a single key "score" and value either "yes" or "no".
Example: {{"score": "yes"}}
Do not include any preamble, explanation, or markdown formatting."""

hallucination_grader_prompt = PromptTemplate(
    template=HALLUCINATION_GRADER_TEMPLATE,
    input_variables=["generation", "documents"],
)


# Answer Usefulness Grader: Checks if answer resolves the question
ANSWER_GRADER_TEMPLATE = """You are a grader assessing whether an answer is useful to resolve a question.

Here is the answer:
-------
{generation}
-------

Here is the question: {question}

Determine if the answer is useful to resolve the question.
Return ONLY valid JSON using double quotes (not single quotes) with a single key "score" and value either "yes" or "no".
Example: {{"score": "yes"}}
Do not include any preamble, explanation, or markdown formatting."""

answer_grader_prompt = PromptTemplate(
    template=ANSWER_GRADER_TEMPLATE,
    input_variables=["generation", "question"],
)


# Question Rewriter: Optimizes questions for better retrieval
QUESTION_REWRITER_TEMPLATE = """You are a question rewriter that converts an input question to a better version optimized for vectorstore retrieval.

Look at the initial question and formulate an improved question that will retrieve more relevant documents.

Here is the initial question: {question}

Improved question with no preamble:"""

question_rewriter_prompt = PromptTemplate(
    template=QUESTION_REWRITER_TEMPLATE,
    input_variables=["question"],
)
