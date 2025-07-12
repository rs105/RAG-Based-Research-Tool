from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

# Create a new customized prompt by prepending our custom instructions to the default template
new_template = (
    "You are a helpful, experienced real-estate assistant agent."
    "Answer the user's question based only on the below context."
    "If the answer is not in the context, just say you don't know.\n\n"
) + template

# Define the main PromptTemplate for the QA chain
# This template will be used to generate the final answer
PROMPT = PromptTemplate(
    template=new_template,
    input_variables=["summaries", "question"]
)

# Define the template for formatting individual document chunks
# This is used when showing retrieved sources to the LLM
EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)