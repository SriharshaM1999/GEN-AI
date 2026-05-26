from langchain_core.prompts import PromptTemplate

# template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input', 'style_input','length_input'],
validate_template=True  # This will check if the template is correctly formatted and all input variables are present in the above template
)

template.save('template.json')



"""
Notes:
1. PromptTemplate is a class that allows us to create a template for our prompts with placeholders for dynamic input variables.

2. The template string includes the instruction the llm has to follow,
 and the input_variables list specifies which variables will be dynamically filled in when we use the template.

3. The validate_template=True, ensures that the template is correctly formatted and that all input variables specified are present in the template string.

4. When you run this file, a template.json file will be created in the same directory,
 which can be loaded and used in other parts of the code to generate prompts dynamically based on user input or other factors.
"""

