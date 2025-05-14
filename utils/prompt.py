SFT_RMA_TRAIN_PHI4 = """<|system|>Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|end|><|user|>{data}<|end|><|assistant|>{answer}<|end|><|endoftext|>"""
SFT_RMA_INFERENCE_PHI4 = """<|system|>Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|end|><|user|>{data}<|end|><|assistant|>"""

SFT_RMA_TRAIN_QWEN3 = """<|im_start|>system
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|im_end|>
<|im_start|>user
{data}<|im_end|><|im_start|>assistant{answer}<|im_end|>"""

SFT_RMA_INFERENCE_QWEN3 = """<|im_start|>system
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. <|im_end|>
<|im_start|>user
{data}<|im_end|><|im_start|>assistant"""

SFT_HISTORY_TRAIN_QWEN25 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}
<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

SFT_HISTORY_INFERENCE_QWEN25 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

SFT_REWRITE_TRAIN_QWEN25 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}
<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

SFT_REWRITE_INFERENCE_QWEN25 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
"""

SFT_HISTORY_INFERENCE_QWEN3 = """<|im_start|>system
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
Conversation History: {conversation_history}
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

SFT_REWRITE_INFERENCE_QWEN3 = """<|im_start|>system
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|im_end|>
<|im_start|>user
User Query: {data}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

SFT_REWRITE_INFERENCE_PHI4 = """<|system|>Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
 <|tool|>{tools}<|/tool|><|end|><|user|>User Query: {data}<|end|><|assistant|>"""

SFT_HISTORY_INFERENCE_PHI4 = """<|system|>You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.
 <|tool|>{tools}<|/tool|><|end|><|user|>Conversation History: {conversation_history}\nUser Query: {data}<|end|><|assistant|>"""

SFT_HISTORY_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
Conversation History: {conversation_history}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

SFT_HISTORY_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
Conversation History: {conversation_history}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SFT_REWRITE_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

SFT_RMA_TRAIN_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>
"""

SFT_RMA_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history.
Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
<|eot_id|><|start_header_id|>user<|end_header_id|>

{data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SFT_REWRITE_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. 
If no tool matches the query, set the tool to "None". 
Only use parameter values that are explicitly stated or can be reasonably inferred from the query.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Tools: {tools}
User Query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

ZERO_HISTORY_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. You may refer to the conversation_history to generate an appropriate response.
7. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text
<|eot_id|><|start_header_id|>user<|end_header_id|>
Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

conversation_history: {conversation_history}
query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

ZERO_REWRITE_INFERENCE_LLAMA = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. output the response strictly as pure JSON in the exact format provided below. Do not include any additional text
<|eot_id|><|start_header_id|>user<|end_header_id|>
Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

query: {data}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

COMPLEX_REWRITE_INFERENCE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.
The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, demonstrative adjectives, or referential expressions (such as "it," "that," "there," "that xml file," "that image file," "that number," "that file uri," "that contact," "that location," "her," "him") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1: Identify all ambiguous pronouns, demonstratives, demonstrative adjectives, or referential terms in the query (including next_turn_query), and explicitly replace each of them with corresponding details from the conversation_history. Especially, replace demonstrative adjectives such as "that xml file," "that image file," "that number," "that file uri," "that contact," "that location," "her," or "him" appearing in next_turn_query explicitly using conversation_history. Try to keep the original query sentence structure whenever possible.
Rule 2: Keep exactly the same sentence structure as the original query. Only replace ambiguous references with explicit values.
Rule 3: If the query references specific values such as URIs, file paths, phone numbers, email addresses, or physical addresses, you MUST explicitly include these values in the rewrited_query. Every explicit value used in the rewrited_query must appear exactly as presented in the conversation_history or query. Do NOT alter these values in any way, and do NOT generate or include any additional information that is not explicitly provided.
Rule 4: Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. No additional context, explanation, or commentary should be added.

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

COMPLEX_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as concise as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided.
 3. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 4. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 6. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using demonstrative adjective (e.g., "that xml file", "that image file", "that number", "that file uri", "that contact", "that location", "her", "him")
 7. Use demonstrative adjective to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 8. Each next_turn_query should have only one purpose.
 9. Do not generate same next_turn_query among the provided data.

Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous demonstrative adjective used in next_turn_query (e.g., "that xml file", "that image file", "that number", "that file uri", "that contact", "that location", "her", "him") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace demonstrative adjective with explicit values only when the demonstrative adjective refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using demonstrative adjective
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains demonstrative adjective.
rule 3. if demonstrative adjective represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but demonstrative adjective
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

ZERO_REWRITE_PROMPT = """
Please write a rewrited_query based on the given conversation_history and query.
The purpose of the rewrited_query is to convert the query into a clearer and more explicit sentence.

Rule 1: Replace ambiguous pronouns, demonstratives, or referential expressions (e.g., "it," "that," "there") in the query with explicit information from the conversation_history.
Rule 2: The rewrited_query should have the same sentence structure as the query, but replace all pronouns, demonstratives, and referential terms with explicit information.
Rule 3: Any explicit values used in the rewrited_query must be taken directly from the conversation_history.
Rule 4: Do not add any new information or values that are not present in the conversation_history or query.

format: {re_format}

now, please write a rewrited_query based on the given conversation_history and query.
Generate only json format data refer to the format above.

conversation_history: {conversation_history}
query: {query}
"""

ZERO_SHOT_SMALL_REWRITE_PROMPT = """
Rewrite the query clearly by replacing ambiguous pronouns (like "it", "that") with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.

Respond ONLY in JSON format without any explanation:
{{"rewrited_query": "<rewritten query>"}}
data: {data}
"""

FEW_SHOTS_SMALL_REWRITE_PROMPT = """
Rewrite the query clearly by replacing ambiguous pronouns emonstrative with explicit information from the conversation history. Keep exactly the same sentence structure. Do NOT generate or include any information, words, or values outside of the provided conversation_history and query.
examples: {examples}

Respond ONLY in JSON format without any explanation:
{{"rewrited_query": "<rewritten query>"}}
data: {data}
"""

FEW_SHOTS_SIMPLER_REWRITE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.

The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, or referential expressions (such as "it," "that," "there") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1. Rewrite the query clearly into a fully self-contained sentence, explicitly replacing ambiguous references (like 'it' or 'that') with details directly from conversation_history.
Rule 2. Keep exactly the same sentence structure.
Rule 3. Include values (e.g., URIs, phone numbers, addresses) exactly as given.
Rule 4. Do NOT add any information not explicitly provided

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

FEW_SHOTS_REWRITE_PROMPT = """
Please generate a rewrited_query based on the provided conversation_history and query.

The goal of the rewrited_query is to rewrite the original query into a clear, explicit, and fully self-contained sentence by replacing ambiguous pronouns, demonstratives, or referential expressions (such as "it," "that," "there") with explicit information directly from the conversation_history.

Important Guidelines:
Rule 1: Identify all ambiguous pronouns, demonstratives, or referential terms in the query and replace each of them explicitly with corresponding details from the conversation_history.
Rule 2: Keep exactly the same sentence structure as the original query. Only replace ambiguous references with explicit values. However, exclude any natural fillers, hesitations, or conversational asides when rewriting the sentence.
Rule 3: If the query references specific values such as URIs, file paths, phone numbers, email addresses, or physical addresses, you MUST explicitly include these values in the rewrited_query. Every explicit value used in the rewrited_query must appear exactly as presented in the conversation_history or query. Do NOT alter these values in any way, and do NOT generate or include any additional information that is not explicitly provided.
Rule 4: Do NOT generate or include any information, words, or values outside of the provided conversation_history and query. No additional context, explanation, or commentary should be added.

Here are clear and correct examples demonstrating the desired format and behavior:
examples: {examples}

Now, using only the data provided below, generate the rewrited_query. Respond strictly following the JSON format given below and include no additional text or explanation:
{{"rewrited_query": ""}}
data:
{data}
"""

ZERO_SHOT_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

Query: {data}
"""

ZERO_SHOT_HISTORY_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for the query and assist in generating the parameters. I will provide descriptions of several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query implicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.
6. You may refer to the conversation_history to generate an appropriate response.

Below is the information on the tools I provided:
Tools: {tools}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key which include tool and parameters inside. 
format: {re_format}
The result should be in JSON format, following the same structure as the format above.    
Do not create any parameter values that cannot be inferred from the user’s query.

conversation_history: {conversation_history}
Query: {data}
"""

FEW_SHOT_INFERENCE_GENERATION_PROMPT = """
Please help identify the appropriate plan for a given query and assist in generating its parameters.  
I will provide descriptions and example data for several tools.  
Based on these tools, you must generate a response to the given query. In other words, the response should be written in a way that calls the appropriate tool to fulfill the user's request. The requirements are as follows:

1. For each query, you must choose the appropriate tool from the given list and provide a response. This means you must specify the value of each parameter used in the selected tool.
2. If the query does not match any of the provided tools, set the plan to "None".
3. If a parameter has `required=False`, you may omit it. However, if the query explicitly includes information corresponding to that parameter, you must include its value.
4. The generated data must follow the format of the example provided for that tool.
5. All parameter values used in the plan call must be inferable from the user's request. Do not make up values that are not present or implied in the query.

Below are examples of tool descriptions and query-response pairs that I’ve provided:  
Tool-example pairs: {tools-examples}

Now, please generate a response to the query below. The generated data must include both the query and the correct answer key.  
The result should be in JSON format, following the same structure as the examples above.  
Do not create any parameter values that cannot be inferred from the user’s query.
{data}
"""

RESPONSE_GENERATION_PROMPT = """
After successfully executing a plan based on the user's query, this will help generate natural and accurate device_responses for the mobile AI assistant. You will receive detailed tool descriptions (plans), results from plan execution, previous conversation history, and the user's queries. Your task is to explicitly create clear, contextually appropriate device_responses based on these inputs.

Carefully follow the instructions below:
 1. Diversity and Successful device_response (Brief): Utilize diverse vocabulary and sentence structures to maintain naturalness with varying tones and attitudes, yet keep concise.
 2. Contextual Accuracy: Fully understand the provided tool description and user query to generate contextually accurate device_responses. 
 3. When generating a device_response for tools like VIEW_SENT_SOMETHINGS that check past records, ensure the reliability of the data by referencing the conversation_history that contains past device_response records.
 4. Please refer to the 'returns' section in the tool description for the format of the content to be written in device_response. Based on the 'returns' section, generate a variety of examples in a similar format. If conversation_history exists, you may also need to refer to previous responses.\
 5. If the return value is not explicitly stated in the user query but needs to be generated creatively, it should be produced in a diverse and non-repetitive manner.
 6. In addition, make sure to explicitly generate device_responses that clearly reflect the values returned by the executed plan. If the returned values include specific details such as URIs or phone numbers, state them clearly and accurately. 
 
Below is the provided tool (plan) including the plan and execution results:
Tool: {tool}

When generating a device_response based on the tool, each conversation_history, and query, imagine yourself as a Mobile Device. Also, make sure to vary the tone and phrasing for each generated response.
Based on the provided query-device_response sets below, clearly and appropriately add a 'device_response' field.
Do not modify any of the values provided in the query-device_response sets. Please add a new key "device_response" with an appropriate value, directly to the top-level of the JSON object, not inside any nested field
{dataset}
"""

MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request that fits the context.
 2. The queries should be as simple as possible. also a query can be a command, a question, or a request with detailed descriptions, etc. never explaining about it.
 3. The generated queries should comprehensively cover all possible uses of the tool, ensuring the coverage of various parameters, thus verifying the tool's ability to handle various forms of work.
 4. The generated queries should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make.
 6. When objects or return values are included in the 'query' and 'device_response', you must utilize them, but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him." in case that, Every next_turn_query must use pronouns.
 7. If pronouns in next_turn_query do not indicate the required parameter, next_turn_query must be generated as a sentence that includes the required parameter.

Guidelines for writing rewrited_query:
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn.
 2. If the user's request is already clear and specific, rewrite_query should be same as next_turn_query
 3. Also If next_turn_query has not pronouns, rewrite_query should be same as next_turn_query
 4. If the next_turn_query contains pronouns and device_response contains specific values, the rewrited_query must explicitly use those specific values.
 5. The rewrited_query should be a clear and easily understandable sentence.

[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query fields to the all JSON datas below. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible.

[previous_turn_datas]
{previous_turn_data}
"""

SIMPLE_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as concise as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided.
 3. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 4. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 5. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 6. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him.". 
 7. Use pronouns to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 8. Each next_turn_query should have only one purpose.
 9. Do not generate same next_turn_query among the provided data.
 
Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace pronouns with explicit values only when the pronouns refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using pronouns
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains pronouns.
rule 3. if pronouns represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but pronouns
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

DIFFICULTY_MULTI_TURN_GENERATION_PROMPT = """
Below are the user's previous questions and the system's responses in [previous_turn_datas], along with a description of the next possible action in [description].
Based on [previous_turn_datas] and [description], you need to write a natural and clear follow-up request from the user's perspective, named next_turn_query and explicitly rewrite this as rewrited_query.

Guidelines for writing next_turn_query:
 1. Clearly understand the content of the previous turn (query and device_response) and create a natural follow-up request like human that fits the context.
 2. The next_turn_query should be as complicated as possible and can be written in various forms, such as commands, questions, or asking with detailed descriptions. However, explanations about the next_turn_query themselves should not be provided. 
 3. It should sound as if a real person is talking to a device—include natural fillers, hesitations, or even unnecessary remarks to make it feel more human.
 4. The generated next_turn_query, based on the query and device_response, should comprehensively cover all possible use cases of the tool, including various parameters to verify the tool's ability to handle different types of tasks. Information for parameters with required=True must be included.
 5. The generated next_turn_query should be solvable using the given tools. if not, next_turn_query should be blank value like ""
 6. Use the provided [description] explanations (especially refer 'reason' key for the next-turn query) to naturally and creatively express requests that the user might realistically make. 
 7. When objects or return values are included in the 'query' and 'device_response', you should utilize them only in case that represent provided tool's parameter. but avoid explicitly mentioning their values directly. Instead, refer to them using pronouns such as "it," "that," "there," "her," or "him.". 
 8. Use pronouns to refer to one or more of the provided tool's parameters by utilizing conversation_history, query or device_response.
 9. Each next_turn_query should have only one purpose.
 10. Do not generate same next_turn_query among the provided data.
 
Guidelines for writing rewrited_query:
The purpose of the rewrited_query is to convert the next_turn_query into a clear sentence.
 1. Replace ambiguous pronouns used in next_turn_query (e.g., "it," "that," "there") with explicit information (query or device_response) from the previous turn. do not change sentence format against multi_turn_query
 2. explicit value in rewrite_query is take it from query, device_response, next_turn_query
 3. Keep the structure and form of the next_turn_query and sentence as they are, but replace pronouns with explicit values only when the pronouns refer to conversation_history, query, or device_response. 

Again, you need to consider important rules below.
rule 1. next_turn_query covers required parameter in description using pronouns
rule 2. If conversation_history, query, and device_response include explicit values representing the provided tool parameters, then next_turn_query should contains pronouns.
rule 3. if pronouns represent provided tool's paramter, rewrited_query is same format sentence with next_turn_query, but pronouns
rule 4. explicit value in rewrite_query is take it from query, device_response, next_turn_query 
 
[description]
{description}

[example]
{example}

Without modifying the JSON data below, add the next_turn_query, rewrited_query with an appropriate value, directly to the top-level of the JSON object, not inside any nested field. Then generate the entire JSON data for all. Please generate next_turn_query as diversely as possible and it should align with the intended purpose of the tool and be something that a real user would actually say or use.

[previous_turn_datas]
{previous_turn_data}
"""

REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT = """
Please help me generate a plan-calling dataset. I will provide a tool description and some example data. 
Based on this tool, you need to generate answers for the provided queries. In other words, you need to provide answers that call the tool to resolve the user's query. Here are my requirements:

1. For the generated queries, you must provide answers using the tool. That is, specify the tool used and the values for each parameter.
2. When providing parameters, if a parameter has required=False, it is not necessary to provide its value. However, if the query includes that parameter, you must provide it.
3. The generated data must be presented in the format given in the example above.
4. The parameter values generated with plan calls must be values that can be inferred from the user's query. You must not fabricate parameters that cannot be obtained from the user's request.
5. If a query does not include a required=True parameter or does not provide enough information, set the answer to 'N/A'."

Below are the tool I provided and some examples of query-answer pairs:
Tool: {tool}
Examples: {examples}

Now, please help me generate answer pairs for the queries below. The generated data should include both a query and an answer key.
The results must be generated in JSON format like the example above.
Do not fabricate parameters for the tool; only use parameters that can be inferred from the user's request.
{data}
"""

REWRITED_MULTI_TURN_DATA_GENERATION_PROMPT = """
Please help me generate a plan-calling dataset. I will provide a tool description and some example data. 
Based on this tool, you need to generate answers for the provided queries. In other words, you need to provide answers that call the tool to resolve the user's query. Here are my requirements:

1. For the generated queries, you must provide answers using the tool. That is, specify the tool used and the values for each parameter.
2. When providing parameters, if a parameter has required=False, it is not necessary to provide its value. However, if the query includes that parameter, you must provide it.
3. The generated data must be presented in the format given in the example above.
4. The parameter values generated with plan calls must be values that can be inferred from the user's query. You must not fabricate parameters that cannot be obtained from the user's request.
5. If a query does not include a required=True parameter, set the answer.plan value as 'N/A' and answer.arguments.reason with reasons"

Below are the tool I provided and some examples of query-answer pairs:
Tool: {tool}
Examples: {examples}

Now, please help me generate answer pairs for the queries below. The generated data should include both a query and an answer key.
The results must be generated in JSON format like the example above.
Do not fabricate parameters for the tool; only use parameters that can be inferred from the user's request.
{data}
"""

ITERATION_SIMPLE_FILTERING_PROMPT = """
You are an expert evaluator tasked with reviewing generated training data for AI model training. Your task is to verify the quality of each provided data example according to the following criteria:
Evaluate the provided data based on these criteria:

1. English Language:
 * Verify if all content in the data (conversation_history, query, rewrited_query, answer) is written exclusively in English.
 * If this criterion is not met, explain briefly under the key reason.

2. API Rule Compliance:
 * Confirm that the parameters provided in answer.arguments correctly match the expected parameter types described for the respective API action.
 * For example, if a parameter expects a physical address but receives an email address, this constitutes a failure. Explicitly highlight any such discrepancies under reason.

Your response must be structured in JSON format:
{{
  "English Language": {{"status": "Pass" or "Fail", "reason": "If fail, describe why"}},  
  "API Rule Compliance": {{"status": "Pass" or "Fail", "reason": "If fail, describe why and specify the mismatched parameter"}}
}}

Carefully follow these instructions and ensure clarity and accuracy in your evaluation.
{data}
"""

COMPLEX_INSTRUCTION_SEED_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. THE PARAMETER VALUES GENERATED WITH FUNCTION CALL GENERATED MUST BE VALUES THAT CAN BE INFERRED FROM THE USER'S QUERY; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. THE GENERATED QUERY SHOULD CONTAIN ENOUGH INFOMATION SO THAT YOU COULD CORRECTLY GENERATE PARAMETER USED BY THE TOOLS. THIS IS
 ALSO TO GUARANTEE THAT YOU DON'T FABRICATE PARAMETERS.
10. You should use all the tools I provided to generate the query and answer. It means that you should generate a query that needs to use all the tools I provided to solve, and remember to provider an answer that uses all the tools to solve the query.
11. You can use the same tool multiple times in a single query to ensure the query diversity.
12. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.
13. Generate data of nested function calls if possible. i.e., the argument of a function call is the response of another function call.

following are some examples:
$examples
tools: [
  {
    "name": "send_email",
    "description": "Send an email to a certain mailbox.",
    "arguments": {
       "to": {
         "description": "email address to send to",
         "type": "str",
         "required": true
       }
    }
  },
  {
    "name": "get_email_by_name",
    "description": "get email by the name searching contacts and return the email",
    "arguments": {
      "name": {
        "description": "name to find the email",
        "type": "str",
        "required": true
      }
    },
    "returns": {
      "decription": "Return the email address of the contact person corresponding to the name."
      "type": "str"
    }
  }
]
response: 
{
  "query": "Help me send an email to Tom",
  "answers": [
    {
      "id": 0,
      "name": "get_email_by_name",
      "arguments": {
        "name": "Tom",
      }
    }, 
    {
      "id": 1,
      "name": "send_email",
      "arguments": {
        "to": "#1"
      }
    }
  ]  
}

tools: [
  {
    "name": "dial",
    "description": "Opens the dialer with a specified number in a phone app for user.\n\nThis function helps user to start a phone call process. It can open\nthe dialer with a pre-filled number. User can then choose to dial the number.",
    "arguments": {
      "phone_number": {
        "description": "The phone number to dial. This should be a valid\ntelephone number as defined in IETF RFC 3966. Examples include:\n\"2125551212\" or \"(212) 555 1212\".",
        "type": "str",
        "required": true
      }
    },
    "examples": [
      "# Open dialer with a number\ndial(\"2125551212\")"
    ]
  },
  {
    "name": "get_contact_info",
    "description": "Get the contact information based on the contact name and the key.\n",
    "arguments": {
      "name": {
        "description": "The name of the contact.",
        "type": "str",
        "required": true
      },
      "key": {
        "description": "The key to get the information of the contact.\ncan be one of the following: \"email\", \"phone\", \"address\" \"uri\"\nif key is \"uri\", this function will return the uri of the contact that can be \nused to edit the contact.",
        "type": "str",
        "required": true
      }
    },
    "returns": {
      "description": "The information of the contact based on the key.",
      "type": "str"
    },
    "examples": [
      "get_contact_info(\"John Doe\", \"email\")\nthis will return the email of the contact named \"John Doe\""
    ]
  }
]
responese:
{
  "query": "I want to call my friend Victor",
  "answers": [
    {
      "id": 0,
      "name": "get_contact_info",
      "arguments": {
        "name": "Victor",
        "key": "phone"
      }
    },
    {
      "id": 1,
      "name": "dial",
      "arguments": {
        "phone_number": "#0"
      }
    }
  ]
}

Now I will give you a tool, and you help me generate 15 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE AND PUT IT IN A JSON LIST.
REMEMBER YOU SHOULD USE ALL THE TOOLS AT ONE QUERY AND SOLVE IT WITH ALL TOOLS, AND GENERATE NESTED CALL IF POSSIBLE.
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
tools: 
$tools
"""

COMPLEX_INSTRUCTION_GEN_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. THE PARAMETER VALUES GENERATED WITH FUNCTION CALL GENERATED MUST BE VALUES THAT CAN BE INFERRED FROM THE USER'S QUERY; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. THE GENERATED QUERY SHOULD CONTAIN ENOUGH INFOMATION SO THAT YOU COULD CORRECTLY GENERATE PARAMETER USED BY THE TOOLS. THIS IS
 ALSO TO GUARANTEE THAT YOU DON'T FABRICATE PARAMETERS.
10. You should use all the tools I provided to generate the query and answer. It means that you should generate a query that needs to use all the tools I provided to solve, and remember to provider an answer that uses all the tools to solve the query.
11. You can use the same tool multiple times in a single query to ensure the query diversity.
12. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.
13. Generate data of nested function calls if possible. i.e., the argument of a function call is the response of another function call.


Now I will give you some tools and some example data of query-answer pairs using these tools. 
Please help me generate 40 query-answer pairs.
tools: $tools
examples: $examples

REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE AND PUT IT IN A JSON LIST.
REMEMBER YOU SHOULD USE ALL THE TOOLS AT ONE QUERY AND SOLVE IT WITH ALL TOOLS, AND GENERATE NESTED CALL IF POSSIBLE.
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
"""

SEED_GENERATION_PROMPT_ = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate english queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. The parameter values generated with function call generated must be values that can be inferred from the user's query; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 15 english query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
tool: $tool
"""

SEED_GENERATION_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate english queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. The parameter values generated with function call generated must be values that can be inferred from the user's query; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 15 english query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
tool: $tool
"""

DATA_GENERATION_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description and some example data for you. 
You need to generate english queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, it is not necessary to provide its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.

following are tool I provided and some examples of english query-answer pairs:
tool: $tool
examples: $examples

Now please help me generate 40 english query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
"""

SYSTEM_PROMPT_FOR_FUNCTION_CALLING = """
You are an expert in composing functions. You are given a query and a set of possible functions. 
Based on the query, you will need to make one or more function calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. Remember you should not use functions that is not suitable for the query and only return the function call in tools call sections. 
"""

SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING = """
You are an expert in composing functions.
"""

JSON_NESTED_CALLING_PROMT = """
If an argument is a response from a previous function call, you can reference it in the following way like the argument value of arg2 in func1:
[
    {
      "id": 0,
      "name": "func0",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    {
      "id": 1,
      "name": "func1",
      "arguments": {
          "arg1": "value1",
          "arg2": "#0",
          ...
      }
    },
    ...
]
This means that the value of arg2 in func1 is the return value from func0 (#0 means the response from the function call with id 0).
"""

CODE_NESTED_CALLING_PROMPT = """
You can do nested function calling in the following way:
result1 = func0(arg1="value1", arg2="value2", ...)
result2 = func1(arg1="value1", arg2=result1, ...)
...
This means that the value of arg2 in func1 is the return value from func0.
"""

JSON_CALL_FORMAT = """
[
    {
      "id": 0,
      "name": "func0",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    {
      "id": 1,
      "name": "func1",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    ...
]
"""

CODE_CALL_FORMAT = """
result1 = func0(arg1="value1", arg2="value2", ...)
result2 = func1(arg1="value1", arg2=result1, ...)
...
"""

FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL = """
Here is a list of functions that you can invoke:
$functions

Should you decide to return the function call(s), Put it in the format of 
$call_format

$nest_prompt

$example

If there is a way to achieve the purpose using the given functions, please provide the function call(s) in the above format.
REMEMBER TO ONLY RETURN THE FUNCTION CALLS LIKE THE EXAMPLE ABOVE, NO OTHER INFORMATION SHOULD BE RETURNED.

Now my query is: $user_query
"""

SHORT_FUNCTION_CALLING_PROMPT = """
Here is a list of functions:
$functions

$call_format

$nest_prompt

$example

Now my query is: $user_query
"""

ANNOTATION_PROMPT = """
Now I have some functions described in JSON format, and I have some query and solution that using these functions.
I use GPT to generate solution for the query, and I need to validate the solution with the given solution.
Some arguments in the functinos should be strictly matched with the given solution, while others should be semantically matched.
I need to annotate the field type of each argument in the functions.
Blow is an example function
{
	"name": "ACTION_INSERT_EVENT",
	"description": "Add a new event to the user's calendar.\n",
	"arguments": {
		"TITLE": {
			"description": "The event title.",
			"type": "str",
			"required": true
		},
		"DESCRIPTION": {
			"description": "The event description.",
			"type": "str",
			"required": true
		},
		"EVENT_LOCATION": {
			"description": "The event location.",
			"type": "str",
			"required": true
		},
		"EXTRA_EVENT_ALL_DAY": {
			"description": "A boolean specifying whether this is an all-day event. Default is False.",
			"type": "bool",
			"required": false,
			"default": false
		},
		"EXTRA_EVENT_BEGIN_TIME": {
			"description": "The start time of the event in ISO 8601 format. Default is None.",
			"type": "str",
			"required": false,
			"default": null
		},
		"EXTRA_EVENT_END_TIME": {
			"description": "The end time of the event in ISO 8601 format. Default is None.",
			"type": "str",
			"required": false,
			"default": null
		},
		"EXTRA_EMAIL": {
			"description": "A list of email addresses that specify the invitees. Default is None.",
			"type": "List[str]",
			"required": false,
			"default": null
		}
	}
}

{
    "TITLE": {
        "reason": "It is no need to generate a title strictly matched with the given solution, so it should be semantically matched.",
        "match_type": "semantic",
    },
    "DESCRIPTION": {
        "reason": "The description is subjective and is uncertain. So it should be semantically matched.",
        "match_type": "semantic",
    },
    "EVENT_LOCATION": {
        "reason": "Location is a common field, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_ALL_DAY": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_BEGIN_TIME": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_END_TIME": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EMAIL": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    }
}


You should only annotate the arguments, not other fields. If arguments is an empty dict {} you can just annotate
it with an empty dict {}.

Here is the function I give you:
$function

Please annotate the argument field type of the function JUST LIKE THE EXAMPLE ABOVE.
"""
