from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain

from langchain.chains import LLMChain
from langchain.llms import AzureOpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

#import os
#os.environ["OPENAI_API_TYPE"] = "azure"
#os.environ["OPENAI_API_BASE"] = "https://aoai-dv3-api.openai.azure.com/"
#os.environ["OPENAI_API_VERSION"] = "2022-12-01"
#os.environ["OPENAI_API_KEY"] = "7aea0c78fe0f4d74ae3ae94e28a59c07" #os.getenv("OPENAI_API_KEY")
#stream_llm = AzureOpenAI(deployment_name="aoai-dv3", model_name="text-davinci-003", streaming=True, temperature=0.0, max_tokens=5000)

stream_llm = OpenAI(
        streaming=True,
        verbose=True,
        temperature=0,
)

template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{chat_history}
Human: {question}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "question"], 
    template=template
)

def get_chat_chain(
    question_handler, stream_handler, tracing: bool = False
) -> LLMChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        stream_manager.add_handler(tracer)

    #streaming_llm = AzureOpenAI(deployment_name="aoai-dv3", model_name="text-davinci-003", streaming=True, callback_manager=stream_manager, temperature=0.0, max_tokens=5000)
    streaming_llm = OpenAI(callback_manager=stream_manager, verbose=True, streaming=True, temperature=0.0, max_tokens=1000)

    chatgpt_chain = ConversationChain(
        memory=ConversationBufferWindowMemory(k=100, return_messages=True, memory_key="chat_history"),
        callback_manager=stream_manager,
        llm=streaming_llm,
        prompt=prompt, 
        verbose=True, 
        input_key="question",
        output_key="answer"
    )

    return chatgpt_chain


