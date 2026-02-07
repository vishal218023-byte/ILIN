import json
import logging
from typing import List, Dict, Any, Optional, Iterator
import requests
from dataclasses import dataclass

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


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        self.base_url = base_url or config.ollama_base_url
        self.model = model or config.ollama_model
        self.temperature = temperature if temperature is not None else config.get('ollama.temperature', 0.7)
        self.max_tokens = max_tokens or config.get('ollama.max_tokens', 2048)
        self.timeout = timeout or config.get('ollama.timeout', 120)
        self.context_window = config.get('ollama.context_window', 4096)
        
        self.rag_template = self._get_rag_template()
    
    def _get_rag_template(self) -> str:
        return """You are an intelligent assistant for the ILIN system (Integrated Localized Intelligence Node). 
Your task is to provide accurate, helpful answers based on the provided context from documents.

CONTEXT INFORMATION:
{context}

USER QUESTION:
{question}

Instructions:
- Answer the question using ONLY the information provided in the context above.
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the available documents."
- Be concise and direct in your response.
- Cite the source documents when providing information.
- If the question asks for guidance or help, provide clear step-by-step instructions when available.

Your response:"""

    def _format_context(self, contexts: List[RAGContext]) -> str:
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            source_info = f"[Source {i}: {ctx.source}, Score: {ctx.score:.3f}]"
            formatted.append(f"{source_info}\n{ctx.content}\n")
        
        return "\n".join(formatted)

    def _count_tokens_approx(self, text: str) -> int:
        return len(text.split())
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False
    ) -> ChatResponse:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                data = response.json()
                return ChatResponse(
                    content=data.get('response', ''),
                    sources=[],
                    model=self.model,
                    done=True
                )
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
    
    def _handle_streaming_response(self, response) -> Iterator[ChatResponse]:
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    yield ChatResponse(
                        content=data.get('response', ''),
                        sources=[],
                        model=self.model,
                        done=data.get('done', False)
                    )
                except json.JSONDecodeError:
                    continue
    
    def chat_with_rag(
        self,
        question: str,
        contexts: List[RAGContext],
        stream: bool = False
    ) -> ChatResponse:
        max_context_tokens = self.context_window - self._count_tokens_approx(question) - 500
        
        filtered_contexts = []
        current_tokens = 0
        
        for ctx in contexts:
            ctx_tokens = self._count_tokens_approx(ctx.content)
            if current_tokens + ctx_tokens > max_context_tokens:
                break
            filtered_contexts.append(ctx)
            current_tokens += ctx_tokens
        
        context_str = self._format_context(filtered_contexts)
        prompt = self.rag_template.format(
            context=context_str,
            question=question
        )
        
        sources = [
            {
                'document_id': ctx.document_id,
                'source': ctx.source,
                'score': ctx.score,
                'excerpt': ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
            }
            for ctx in filtered_contexts
        ]
        
        if stream:
            return self._stream_rag_response(prompt, sources)
        else:
            response = self.generate(prompt, stream=False)
            response.sources = sources
            return response
    
    def _stream_rag_response(
        self,
        prompt: str,
        sources: List[Dict[str, Any]]
    ) -> Iterator[ChatResponse]:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        yield ChatResponse(
                            content=data.get('response', ''),
                            sources=sources if data.get('done', False) else [],
                            model=self.model,
                            done=data.get('done', False)
                        )
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error in streaming RAG: {str(e)}")
            raise
    
    def check_model_available(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            
            return self.model in models or any(self.model in m for m in models)
            
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []


ollama_client = OllamaClient()
