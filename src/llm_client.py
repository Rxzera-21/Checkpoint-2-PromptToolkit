"""
llm_client.py — Aula 05
Conexão com Ollama API local.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


class LLMClient:
    """Cliente para comunicação com a API REST do Ollama."""

    def __init__(self, host: str = None, model: str = None):
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.api_url = f"{self.host}/api/chat"

    def chat(
        self,
        prompt: str,
        system: str = "",
        temp: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        """
        Envia um prompt ao LLM e retorna resposta com métricas.

        Returns:
            dict com chaves: resposta, tokens_prompt, tokens_resposta, tempo_ms
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temp,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                tempo_ms = int((time.time() - t0) * 1000)

                if response.status_code != 200:
                    raise ValueError(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )

                data = response.json()
                resposta = data.get("message", {}).get("content", "").strip()

                # Contar tokens via Ollama (eval_count)
                tokens_prompt = data.get("prompt_eval_count", 0)
                tokens_resposta = data.get("eval_count", 0)

                return {
                    "resposta": resposta,
                    "tokens_prompt": tokens_prompt,
                    "tokens_resposta": tokens_resposta,
                    "tokens_total": tokens_prompt + tokens_resposta,
                    "tempo_ms": tempo_ms,
                }

            except requests.exceptions.Timeout:
                print(f"  [AVISO] Timeout na tentativa {attempt}/{MAX_RETRIES}.")
                if attempt == MAX_RETRIES:
                    return self._erro("Timeout após todas as tentativas.")
                time.sleep(2 * attempt)

            except requests.exceptions.ConnectionError:
                print(
                    f"  [ERRO] Não foi possível conectar ao Ollama em {self.host}."
                )
                print(
                    "  Verifique se o Ollama está rodando: ollama serve"
                )
                return self._erro("Conexão recusada. Ollama não está rodando?")

            except Exception as e:
                print(f"  [ERRO] Tentativa {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    return self._erro(str(e))
                time.sleep(1)

    @staticmethod
    def _erro(msg: str) -> dict:
        return {
            "resposta": f"[ERRO: {msg}]",
            "tokens_prompt": 0,
            "tokens_resposta": 0,
            "tokens_total": 0,
            "tempo_ms": 0,
        }

    def health_check(self) -> bool:
        """Verifica se o Ollama está rodando."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

