"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from langchain.callbacks.base import AsyncCallbackHandler
from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from chat import get_chat_chain
from legal import get_memo_chain, write_memo
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

vs_path = "vectorstore_law.pkl"

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path(vs_path).exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open(vs_path, "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

@app.get("/chatbot")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/law")
async def get(request: Request):
    return templates.TemplateResponse("index_law.html", {"request": request})

@app.get("/memo")
async def get(request: Request):
    return templates.TemplateResponse("index_memo.html", {"request": request})

@app.websocket("/chatbot")
async def websocket_chatbot(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    #for free form chat with GPT
    qa_chain = get_chat_chain(question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

@app.websocket("/law")
async def websocket_chatbot(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    #for Q&A over a database
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)

    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))
            end_resp = ChatResponse(sender="bot", message="\n\n"+str(result["source_documents"]), type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


@app.websocket("/memo")
async def websocket_chatbot(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    #for Q&A over a database
    qa_chain = get_memo_chain(question_handler, stream_handler, tracing=False)
     
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            jsonIn = await websocket.receive_json()
            
            if (jsonIn["id"] == "memo") :
                # Construct a response  
                statement = jsonIn["content"][0]
                law = jsonIn["content"][1]
                question = jsonIn["content"][2]
                caution = jsonIn["content"][3]
                start_resp = ChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())

                result = await qa_chain.arun(statement=statement, question=question, law=law, caution=caution)
        
                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

#Post API for law research
#from typing import Any, Callable, Dict, List, Optional, Tuple, Union
#CHAT_TURN_TYPE = Union[Tuple[str, str], str]    
#chat_history: List[CHAT_TURN_TYPE]
@app.post("/research")
async def post_research(statement: str, question: str, chat_history: Optional[str]=None):
    qa_chain = get_chain(vectorstore, AsyncCallbackHandler, AsyncCallbackHandler)
    result = await qa_chain.acall(
                {"question": statement+question, "chat_history": chat_history}
            )
    return result

#Post API for Memo generation
@app.post("/memo")
async def post_memo(statement: str, question: str, law: str, good: Optional[str]=None, bad: Optional[str]=None):
    memo = await write_memo(
        statement, question, law, good, bad
        )
    return memo
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
