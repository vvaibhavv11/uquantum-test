from config import settings
import httpx

class LLMService:
    def __init__(self):
        self.default_groq_key = settings.GROQ_API_KEY
        self.groq_base_url = "https://api.groq.com/openai/v1"
        self.openai_base_url = "https://api.openai.com/v1"
        self.google_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"

    def _get_provider_info(self, model: str, api_keys: dict = None):
        """
        Determines the provider, base URL, and API key for a given model.
        """
        model_lower = model.lower()
        api_keys = api_keys or {}
        
        # 1. Check for Google Gemini
        if "gemini" in model_lower:
            key = api_keys.get("google")
            return "google", self.google_base_url, key
        
        # 2. Check for OpenAI
        if "gpt" in model_lower:
            key = api_keys.get("openai")
            return "openai", self.openai_base_url, key
            
        # 3. Default to Groq
        key = api_keys.get("groq") or self.default_groq_key
        return "groq", self.groq_base_url, key

    async def generate_completion(self, messages: list, model: str = "llama-3.1-8b-instant", mode: str = "ask", api_keys: dict = None):
        """
        Generates a completion using the appropriate LLM provider.
        """
        provider, base_url, api_key = self._get_provider_info(model, api_keys)
        
        if not api_key:
            return {"error": f"No API key found for provider {provider}. Please configure it in Settings."}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Determine the system prompt based on the mode
        if mode == "code":
            system_content = (
                "You are an expert Quantum Coding Assistant. Your primary goal is to provide high-quality, "
                "production-ready quantum code. \n"
                "1. AGENTIC CODING FLOW: Break down complex coding tasks into logical steps. "
                "Explain the architecture before providing the implementation.\n"
                "2. QISKIT 1.0+: Always use version 1.0+ syntax (e.g., 'qiskit_aer', 'backend.run').\n"
                "3. OPTIMIZATION: Suggest ways to optimize circuits for depth, width, or noise resilience.\n"
                "4. FORMATTING: Use markdown blocks with language tags (```python, ```qasm, etc.).\n"
                "5. NO FLUFF: Keep explanations technical and concise."
            )
        else: # 'ask' mode
            system_content = (
                "You are a Senior Quantum Research Scientist. Your goal is to provide deep, detailed analysis "
                "of quantum computing concepts, algorithms, and research papers.\n"
                "1. AGENTIC ANALYSIS: Use a step-by-step reasoning approach. Analyze the problem from multiple "
                "perspectives (mathematical, physical, and computational).\n"
                "2. MODERN STANDARDS: Use recent scientific terminology and reference current industry benchmarks.\n"
                "3. CLARITY: Use LaTeX for math ($...$ or $$...$$) and provide clear analogies where helpful.\n"
                "4. STRUCTURE: Use headings, bullet points, and bold text to organize complex information."
            )

        system_prompt = {
            "role": "system",
            "content": system_content
        }

        # Prepend system prompt if not already present or update existing one
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, system_prompt)
        else:
            messages[0] = system_prompt

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1 if mode == "code" else 0.4,
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # Construct final URL
                url = f"{base_url}/chat/completions"
                
                # Use standard OpenAI headers (Authorization: Bearer <KEY>)
                # This is compatible with OpenAI, Groq, and Google's OpenAI-compatible endpoint
                response = await client.post(
                    url, 
                    json=payload, 
                    headers=headers,
                    timeout=60.0
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    provider_name = provider.capitalize()
                    return {"error": f"{provider_name} API Error ({response.status_code}): {response.text}"}
            except Exception as e:
                return {"error": f"Connection Error: {str(e)}"}

llm_service = LLMService()



