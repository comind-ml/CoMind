from pydantic import BaseModel, SecretStr, Field

class LLMConfig(BaseModel):
    model_name: str = Field(..., description="The name of the LLM model to use")
    api_key: SecretStr = Field(..., description="The API key for the LLM model")
    temperature: float = Field(1.0, description="The temperature for the LLM model")
    max_tokens: int = Field(100000, description="The maximum number of tokens to generate")
    timeout: int = Field(10, description="The timeout for the LLM model")
    max_retries: int = Field(3, description="The maximum number of retries for the LLM model")
    native_function_calling: bool = Field(None, description="Whether to use native function calling")
    keep_history: bool = Field(True, description="Whether to keep the history of the LLM model")
