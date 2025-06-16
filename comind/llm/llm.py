from comind.core.config import LLMConfig
from comind.core.logger import logger
from litellm import completion
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from functools import partial

import re
import json
import jsonschema

FUNC_CALL_SUPPORTED_MODELS = [
    'gpt-4o-mini',
    'gpt-4o',
    'o1-2024-12-17',
    'o3-mini-2025-01-31',
    'o3-mini',
    'o3',
    'o3-2025-04-16',
    'o4-mini',
    'o4-mini-2025-04-16',
    'gpt-4.1',
]

@dataclass
class FunctionSpec(DataClassJsonMixin):
    name: str
    json_schema: dict
    description: str

    def __post_init__(self):
        jsonschema.Draft7Validator.check_schema(self.json_schema)
    
    def to_openai_tool_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema
            },
            "strict": True
        }

class InvalidResponseError(Exception):
    pass

class LLM:
    def __init__(
        self, 
        config: LLMConfig,
        **kwargs: Any
    ) -> None:
        self.config = config

        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        use_native_function_calling = config.native_function_calling
        if use_native_function_calling is None:
            use_native_function_calling = self.config.model_name in FUNC_CALL_SUPPORTED_MODELS

        self.use_native_function_calling = use_native_function_calling
        self.history: List[Dict[str, str]] = []

        self.completion = partial(
            completion,
            model=self.config.model_name,
            api_key=self.config.api_key.get_secret_value(),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        json_pattern = re.compile(r'{.*?}', re.DOTALL)
        matches = json_pattern.findall(text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None
    
    def validate_json(self, json_obj: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        try:
            jsonschema.validate(json_obj, schema)
            return True
        except jsonschema.ValidationError:
            return False
    
    def chat(
        self, 
        user_message: str, 
        function_spec: Optional[FunctionSpec] = None
    ) -> str | Dict[str, Any]:
        logger.debug(f"user_message: {user_message}")

        if not self.config.keep_history:
            self.history = []

        if self.use_native_function_calling and function_spec:
            self.history.append({ "role": "user", "content": user_message })

            for _ in range(self.config.max_retries):

                response = self.completion(
                    messages=self.history,
                    functions=[function_spec.to_openai_tool_dict()],
                    function_call=function_spec.name,
                )

                choice = response.choices[0]
                if choice.finish_reason == "function_call":
                    arguments = self.extract_json(choice.message.function_call.arguments) 
                    valid = self.validate_json(arguments, function_spec.json_schema)

                    if valid: 
                        result = {
                            "type": "function_call",
                            "name": function_spec.name,
                            "arguments": json.loads(choice.message.function_call.arguments)
                        }
                        self.history.append({ "role": "assistant", "content": choice.message.content })
                        return result
                
                logger.warn(f"Failed to parse response, retrying...")
                
            raise InvalidResponseError
        
        if function_spec:
            schema_desc = f"The response should be a valid JSON object that matches the following JSON schema, including all the required fields: {function_spec.json_schema}"
            user_message = f"{user_message}\n\n{schema_desc}"
                
        self.history.append({ "role": "user", "content": user_message })

        for _ in range(self.config.max_retries):
            response = self.completion(messages=self.history)

            content = response.choices[0].message.content 

            if function_spec:
                json_obj = self.extract_json(content)
                valid = self.validate_json(json_obj, function_spec.json_schema)

                if valid:
                    self.history.append({ "role": "assistant", "content": content })
                    return json_obj

            else:
                self.history.append({ "role": "assistant", "content": content })
                return content

            logger.warn(f"Failed to parse response, retrying...")

        raise InvalidResponseError



                     
                    

            





    