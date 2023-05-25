from typing import Optional

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

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



qTemplate="""
在仔细理解以下被三重反引号分隔开的事实陈述, 法律条款和问题后，请用英文写一篇关于案情分析的备忘录： 
  事实陈述：'''{statement}''' 
  法律条款：'''{law}''' 
  问题：'''{question}'''
英文的忘录需要包括以下3段：
1. 列举事实陈述中容易纠结的要点
2. 对问题做出简单和清楚的回答：回答必须基于以上法律的内容，而不能涉及其它法律条款
3. 深度和详细的分析相关法律和案情：法律条款为什么适用于陈述中的场景？请列出推理过程遇到的纠结点和做出的假设。
请特别注意以下事项：
'''{caution}'''
"""

async def write_memo(statement: str, question: str, law: str, good: Optional[str]=None, bad: Optional[str]=None):
    stream_llm = OpenAI(
        streaming=True,
        verbose=True,
        temperature=0,
)
    qPrompt = PromptTemplate(input_variables=["law", "statement", "question", "caution"], template=qTemplate)
    streamChain = LLMChain(llm=stream_llm, verbose=True, prompt=qPrompt)
    memo = await streamChain.arun(
        law=law, statement=statement, question=question
        )
    return memo

def get_memo_chain(
    question_handler, stream_handler, tracing: bool = False
) -> LLMChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        #tracer.load_default_session()
        #stream_manager.add_handler(tracer)

    qPrompt = PromptTemplate(input_variables=["law", "statement", "question", "caution"], template=qTemplate)
    #stream_llm = AzureOpenAI(deployment_name="aoai-dv3", model_name="text-davinci-003", streaming=True, callback_manager=stream_manager, temperature=0.0, max_tokens=5000)
    stream_llm = OpenAI(callback_manager=stream_manager, verbose=True, streaming=True, temperature=0.0, max_tokens=1000)

    stream_Chain = LLMChain(llm=stream_llm, verbose=True, prompt=qPrompt, callback_manager=stream_manager)
    
    return stream_Chain

