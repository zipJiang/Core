# ------------------------------------------------------------------------
# Dec 2023: identify checkworthy
# ------------------------------------------------------------------------
CHECKWORTHY_PROMPT = """
Your task is to identify whether texts are checkworthy in the context of fact-checking.
Let's define a function named checkworthy(input: List[str]).
The return value should be a list of strings, where each string selects from ["Yes", "No"].
"Yes" means the text is a factual checkworthy statement.
"No" means that the text is not checkworthy, it might be an opinion, a question, or others.
For example, if a user call checkworthy(["I think Apple is a good company.", "Friends is a great TV series.", "Are you sure Preslav is a professor in MBZUAI?", "The Stanford Prison Experiment was conducted in the basement of Encina Hall.", "As a language model, I can't provide these info."])
You should return a python list without any other words, 
["No", "Yes", "No", "Yes", "No"], with the same order and length as the input list.
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

checkworthy({texts})
"""

SPECIFY_CHECKWORTHY_CATEGORY_PROMPT = """
You are a factchecker assistant with task to identify a sentence, whether it is 1. a factual claim; 2. an opinion; 3. not a claim (like a question or a imperative sentence); 4. other categories.
Let's define a function named checkworthy(input: str).
The return value should be a python int without any other words, representing index label, where index selects from [1, 2, 3, 4].

For example, if a user call checkworthy("I think Apple is a good company.")
You should return 2
If a user call checkworthy("Friends is a great TV series.")
You should return 1
If a user call checkworthy("Are you sure Preslav is a professor in MBZUAI?")
You should return 3
If a user call checkworthy("As a language model, I can't provide these info.")
You should return 4
Note that your response will be passed to the python interpreter, SO NO OTHER WORDS!

checkworthy("{sentence}")
"""


DECONTEXTUALIZE_PROMPT = """Vague references include but are not limited to:
- Pronouns (e.g. "his", "they", "her")
- Unknown entities (e.g., "this event", "the research", "the invention")
- Non-full names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE.
2. Modify the STATEMENT by replacing vague references with the proper entities from the RESPONSE that they are referring to.
3. You MUST NOT change any of the factual claims made by the original STATEMENT.
4. You MUST NOT add any additional factual claims to the original STATEMENT. For example, given the response "Titanic is a movie starring Leonardo DiCaprio," the statement "Titanic is a movie" should not be changed.
5. Before giving your revised statement, think step-by-step and show your reasoning. As part of your reasoning, be sure to identify the subjects in the STATEMENT and determine whether they are vague references. If they are vague references, identify the proper entity that they are referring to and be sure to revise this subject in the revised statement.
6. After showing your reasoning, provide the revised statement and wrap it in a markdown code block."""


RELEVANCY_PROMPT = """In a given RESPONSE, two subjects are considered "Foo" if the RESPONSE contains information that explains how the two subjects are related.

Instructions:
1. The following STATEMENT has been extracted from the broader context of the given RESPONSE to the given QUESTION.
2. First, state the broad subject of the STATEMENT and the broad subject of the QUESTION.
3. Next, determine whether the subject of the STATEMENT and the subject of the QUESTION should be considered Foo, based on the given definition of "Foo."
4. Before showing your answer, think step-by-step and show your specific reasoning.
5. If the subjects should be considered Foo, say "[Foo]" after showing your reasoning. Otherwise show "[Not Foo]" after showing your reasoning."""


NEXT_SEARCH_PROMPT = """Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Your goal is to try to find evidence that either supports or does not \
support the factual accuracy of the given STATEMENT.
3. To do this, you are allowed to issue ONE Google Search query that you think \
will allow you to find additional useful evidence.
4. Your query should aim to obtain new information that does not appear in the \
KNOWLEDGE. This new information should be useful for determining the factual \
accuracy of the given STATEMENT.
5. Format your final query by putting it in a markdown code block.

KNOWLEDGE:
{knowledge}

STATEMENT:
{input}
"""

SAFE_RATING_PROMPT = """Instructions:
1. You have been given a STATEMENT and some KNOWLEDGE points.
2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. \
The STATEMENT does not need to be explicitly supported by the KNOWLEDGE, but \
should be strongly implied by the KNOWLEDGE.
3. Before showing your answer, think step-by-step and show your specific \
reasoning. As part of your reasoning, summarize the main points of the \
KNOWLEDGE.
4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the \
supporting evidence.
5. After stating your reasoning, restate the STATEMENT and then determine your \
final answer based on your reasoning and the STATEMENT.
6. Your final answer should be either "Supported" or \
"Not Supported". Wrap your final answer in square brackets.

KNOWLEDGE:
{parsed_passages}

STATEMENT:
{input}
"""