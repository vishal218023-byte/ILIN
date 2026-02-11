import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Iterator, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGContext:
    content: str
    source: str
    score: float
    document_id: str

@dataclass
class ChatResponse:
    content: str
    sources: List[Dict[str, Any]]
    model: str
    done: bool

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False) -> Any:
        pass

    @abstractmethod
    def chat_with_rag(self, question: str, contexts: List[RAGContext], chat_history: Optional[List[Dict[str, str]]] = None, stream: bool = False) -> Any:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass

    def _get_rag_template(self) -> str:
        return """You are an intelligent assistant for the ILIN system.
Provide a direct, professional, and accurate response to the user's question.

CRITICAL INSTRUCTION: 
1. Use ONLY the information in the CONTEXT below.
2. DO NOT use phrases like "Based on the provided context", "According to the document", or "The context states".
3. Provide the answer directly as if you are a knowledgeable expert speaking naturally.
4. If the context does not contain the answer, simply state: "Your question is out of the knowledge space."
5. Consider the conversation history to maintain context and answer follow-up questions appropriately.

CONVERSATION HISTORY:
{chat_history}

CONTEXT:
{context}

QUESTION:
{question}

Response:"""

    def _format_context(self, contexts: List[RAGContext]) -> str:
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            formatted.append(f"[Source {i}: {ctx.source}]\n{ctx.content}\n")
        return "\n".join(formatted)

    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        if not chat_history:
            return "No previous conversation."
        formatted = []
        for msg in chat_history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        return "\n".join(formatted)

class OllamaClient(LLMClient):
    def __init__(self):
        self.base_url = config.get('ollama.base_url', "http://localhost:11434")
        self.model = config.get('ollama.model', "llama3.2:3b")
        self.temperature = config.get('ollama.temperature', 0.7)
        self.max_tokens = config.get('ollama.max_tokens', 2048)
        self.timeout = config.get('ollama.timeout', 120)

    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
        }
        if system: payload["system"] = system
        
        response = requests.post(url, json=payload, timeout=self.timeout, stream=stream)
        response.raise_for_status()
        
        if stream:
            return self._handle_stream(response)
        data = response.json()
        return ChatResponse(content=data.get('response', ''), sources=[], model=self.model, done=True)

    def _handle_stream(self, response):
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield ChatResponse(content=data.get('response', ''), sources=[], model=self.model, done=data.get('done', False))

    def chat_with_rag(self, question: str, contexts: List[RAGContext], chat_history: Optional[List[Dict[str, str]]] = None, stream: bool = False):
        prompt = self._get_rag_template().format(
            chat_history=self._format_chat_history(chat_history or []),
            context=self._format_context(contexts),
            question=question
        )
        return self.generate(prompt, stream=stream)

    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m['name'] for m in response.json().get('models', [])]
        except: return []

    def check_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def check_model_available(self, model_name: Optional[str] = None) -> bool:
        target = model_name or self.model
        models = self.list_models()
        return target in models or f"{target}:latest" in models

class LocalGGUFClient(LLMClient):
    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Please wait for installation or run: pip install llama-cpp-python")
            
        self.model_path = model_path
        self.n_ctx = config.get('local_llm.context_window', 4096)
        self.n_threads = config.get('local_llm.threads', 4)
        self.llm = Llama(model_path=model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False)
        self.model_name = os.path.basename(model_path)

    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False):
        full_prompt = f"System: {system}\nUser: {prompt}\nAssistant:" if system else f"User: {prompt}\nAssistant:"
        
        if stream:
            return self._handle_stream(full_prompt)
        
        output = self.llm(full_prompt, max_tokens=config.get('ollama.max_tokens', 2048), stop=["User:", "\nUser:"])
        return ChatResponse(content=output['choices'][0]['text'], sources=[], model=self.model_name, done=True)

    def _handle_stream(self, prompt):
        for chunk in self.llm(prompt, max_tokens=config.get('ollama.max_tokens', 2048), stream=True, stop=["User:", "\nUser:"]):
            text = chunk['choices'][0]['text']
            yield ChatResponse(content=text, sources=[], model=self.model_name, done=False)
        yield ChatResponse(content="", sources=[], model=self.model_name, done=True)

    def chat_with_rag(self, question: str, contexts: List[RAGContext], chat_history: Optional[List[Dict[str, str]]] = None, stream: bool = False):
        prompt = self._get_rag_template().format(
            chat_history=self._format_chat_history(chat_history or []),
            context=self._format_context(contexts),
            question=question
        )
        return self.generate(prompt, stream=stream)

    def list_models(self) -> List[str]:
        models_dir = Path("models")
        if not models_dir.exists(): return []
        return [f.name for f in models_dir.glob("*.gguf")]

class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq python package not installed. Please run: pip install groq")
            
        self.client = Groq(api_key=api_key)
        self.model = model

class NvidiaClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai python package not installed. Please run: pip install openai")
            
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False):
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        if stream:
            return self._handle_stream(messages)
            
        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        return ChatResponse(content=completion.choices[0].message.content, sources=[], model=self.model, done=True)

    def _handle_stream(self, messages):
        stream = self.client.chat.completions.create(model=self.model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            yield ChatResponse(content=content, sources=[], model=self.model, done=False)
        yield ChatResponse(content="", sources=[], model=self.model, done=True)

    def chat_with_rag(self, question: str, contexts: List[RAGContext], chat_history: Optional[List[Dict[str, str]]] = None, stream: bool = False):
        prompt = self._get_rag_template().format(
            chat_history=self._format_chat_history(chat_history or []),
            context=self._format_context(contexts),
            question=question
        )
        return self.generate(prompt, stream=stream)

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Error fetching Nvidia models: {str(e)}")
            return [
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "meta/llama-3.1-405b-instruct",
                "meta/llama-3.1-70b-instruct",
                "meta/llama-3.1-8b-instruct"
            ]

    def generate(self, prompt: str, system: Optional[str] = None, stream: bool = False):
        messages = []
        if system: messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        if stream:
            return self._handle_stream(messages)
            
        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        return ChatResponse(content=completion.choices[0].message.content, sources=[], model=self.model, done=True)

    def _handle_stream(self, messages):
        stream = self.client.chat.completions.create(model=self.model, messages=messages, stream=True)
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            yield ChatResponse(content=content, sources=[], model=self.model, done=False)
        yield ChatResponse(content="", sources=[], model=self.model, done=True)

    def chat_with_rag(self, question: str, contexts: List[RAGContext], chat_history: Optional[List[Dict[str, str]]] = None, stream: bool = False):
        prompt = self._get_rag_template().format(
            chat_history=self._format_chat_history(chat_history or []),
            context=self._format_context(contexts),
            question=question
        )
        return self.generate(prompt, stream=stream)

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"Error fetching Groq models: {str(e)}")
            return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]


class LLMFactory:
    @staticmethod
    def get_client(engine_type: str, **kwargs) -> LLMClient:
        if engine_type == "Ollama":
            return OllamaClient()
        elif engine_type == "Local GGUF":
            return LocalGGUFClient(kwargs.get('model_path'))
        elif engine_type == "Groq (Online)":
            return GroqClient(kwargs.get('api_key'), kwargs.get('model'))
        elif engine_type == "Nvidia (Online)":
            return NvidiaClient(kwargs.get('api_key'), kwargs.get('model'))
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

# Global state for current client
_current_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _current_client
    if _current_client is None:
        # Default to Ollama for now
        _current_client = OllamaClient()
    return _current_client

def set_llm_client(engine_type: str, **kwargs):
    global _current_client
    _current_client = LLMFactory.get_client(engine_type, **kwargs)
