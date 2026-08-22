from typing import List

from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
#from langchain_ollama import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import getpass
import os
os.environ["NVIDIA_API_KEY"] = "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"
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key


@tool
def talk_about_simone() -> bool:
    """ Give information about a person named Simone.
    """
    return "basically he is a great person but with virus in his PC"

@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    #if "Fake St" in addresses or "Pretend Boulevard" in addresses:
    # return True
    if user_id == 123 and "Fake St." in addresses:
        return True
    else:
        return False

@tool
def do_sum(a:int,b:int)->int:
    """Sum two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    """
    return a-b

# Nemotron 3 Nano — efficient reasoning and agentic tasks
llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

#llm = ChatOllama(
#    model="llama3.1:8b",
#    temperature=0.5,
#    num_predict=200,
#    base_url="http://ollama:11434"
#    # other params...
#)

#llm = ChatOllama(
#    model="lfm2.5-thinking:latest",
#    validate_model_on_init=True,
#    temperature=0,
#    base_url="http://ollama:11434"
#).bind_tools([validate_user])

llm=llm.bind_tools([validate_user, talk_about_simone, do_sum])


## NIENTE DIRE DI NON NOMINARE TOOL NON é SUFFICIENTE, MI SA CHE SERVE MODIFICARE L'INOLTRO AL MODELLO DOPO OPPURE FARE UN POST PROCESSING DOPO LA CHIAMATA DEL TOOL
system_prompt = "You are an helpful assistent that can use tools to answer questions." \
"Do not reason too much."

# messages = [
#     ("system", system_prompt),
#     ("human", "Could you validate user 123? They previously lived at 123 Fake St in Boston MA and 234 Pretend Boulevard in Houston TX."),
# ]

# messages = [
#     ("system", system_prompt),
#     ("human", "Could you validate user 123? They previously lived at Fake St."),
# ]

messages = [
    SystemMessage(system_prompt),
    #HumanMessage("Did you know anything about a person named Simone?")
    HumanMessage("Did you know how much is seven plus five?")
]
result = llm.invoke(messages)

#print(result)

# Visualize tool calls made by the model
#for tool_call in result.tool_calls:
    # View tool calls made by the model
#    print(f"Tool: {tool_call['name']}")

# visualize tool calls made by the model
if isinstance(result, AIMessage) and result.tool_calls:
    print(result.tool_calls)

messages.append(result)

##QUA IN TEORIA VANNO FILTRATI E CHIAMATI SOLO I TOOL CHE SERVONO
print(result)
# Step 2: Execute tools and collect results
for tool_call in result.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = do_sum.invoke(tool_call)
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
final_response = llm.invoke(messages)
print(final_response.text)
