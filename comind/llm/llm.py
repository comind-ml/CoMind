from comind.core.config import LLMConfig
from comind.core.logger import logger
from litellm import completion
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin
from functools import partial
from comind.utils.prompt import extract_json
from copy import deepcopy

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
            }
        }

class InvalidResponseError(Exception):
    pass

def _is_empty_value(value: Any) -> bool:
    """
    Check if a value is truly empty (None, empty string, empty list/dict, etc.)
    but not valid falsy values like False or 0.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple)) and len(value) == 0:
        return True
    return False

class LLM:
    def __init__(
        self, 
        config: LLMConfig,
        **kwargs: Any
    ) -> None:
        self.config = deepcopy(config)

        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        use_native_function_calling = config.native_function_calling
        if use_native_function_calling is None:
            use_native_function_calling = self.config.model_name in FUNC_CALL_SUPPORTED_MODELS

        self.use_native_function_calling = use_native_function_calling
        self.history: List[Dict[str, str]] = []

        self._completion = partial(
            completion,
            model=self.config.model_name,
            api_key=self.config.api_key.get_secret_value(),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        self.completion = completion # For testability

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
        logger.debug(f"keep_history: {self.config.keep_history}, user_message: {user_message}")

        if not self.config.keep_history:
            self.history = []

        self.history.append({ "role": "user", "content": user_message })

        if self.use_native_function_calling and function_spec:
            for i in range(self.config.max_retries):
                try:
                    response = self._completion(
                        messages=self.history,
                        tools=[function_spec.to_openai_tool_dict()],
                        tool_choice={"type": "function", "function": {"name": function_spec.name}},
                    )
                    choice = response.choices[0]
                    logger.debug(f"LLM response (native function call, try {i+1}): finish_reason={choice.finish_reason}, message={choice.message}")

                    if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
                        logger.warning(f"Response was not a valid tool call on try {i+1}/{self.config.max_retries}. Finish reason: {choice.finish_reason}. Retrying...")
                        continue

                    tool_call = choice.message.tool_calls[0]
                    arguments = json.loads(tool_call.function.arguments)
                    logger.debug(f"LLM function call arguments (try {i+1}): {arguments}")
                    jsonschema.validate(instance=arguments, schema=function_spec.json_schema)

                    if any(_is_empty_value(v) for v in arguments.values()):
                        logger.warning(f"LLM returned one or more empty values in arguments: {arguments}. Retrying...")
                        continue
                    
                    # Add assistant message with tool_calls to history
                    assistant_message = {
                        "role": "assistant",
                        "content": choice.message.content, # Can be None for tool calls
                        "tool_calls": choice.message.tool_calls
                    }
                    self.history.append(assistant_message)
                    
                    # Add tool response message to complete the conversation flow
                    if self.config.verbose_tool_responses:
                        # Include full JSON, but truncate if too large
                        full_json = json.dumps(arguments)
                        if len(full_json) > 1000:  # Truncate if longer than 1000 chars
                            tool_content = f"{full_json[:500]}...[truncated, total length: {len(full_json)} chars]...{full_json[-200:]}"
                        else:
                            tool_content = full_json
                    else:
                        # Use brief confirmation to save context
                        tool_content = f"Function {function_spec.name} executed successfully with {len(arguments)} parameters."
                    
                    tool_response_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_content
                    }
                    self.history.append(tool_response_message)
                    
                    logger.debug(f"LLM function call successful, returning: {arguments}")
                    return arguments

                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    logger.warning(f"Response validation failed on try {i+1}/{self.config.max_retries}: {e}. Retrying...")
                    continue
            
            raise InvalidResponseError("Failed to get a valid function call response after multiple retries.")
        
        # Fallback for models without native function calling
        if function_spec:
            # Construct a detailed prompt asking for a function call in JSON format
            prompt_addition = (
                "\n\nYou must respond using a function call. "
                f"The function to call is `{function_spec.name}`.\n"
                f"Description: {function_spec.description}\n"
                "The arguments for this function MUST be a single valid JSON object, "
                f"matching this schema: {json.dumps(function_spec.json_schema, indent=2)}\n"
                "Your response must end with a markdown code block containing ONLY the JSON object."
            )
            # This modification happens on a local copy of the message list for this call only
            messages_with_schema = self.history[:-1] + [
                {"role": "user", "content": f"{user_message}{prompt_addition}"}
            ]
        else:
            messages_with_schema = self.history

        for retry_count in range(self.config.max_retries):
            response = self._completion(messages=messages_with_schema)
            content = response.choices[0].message.content
            logger.debug(f"LLM response (fallback, try {retry_count+1}): {content}")

            if function_spec:
                if not content:
                    logger.warning("Response content is empty, retrying...")
                    continue
                json_obj = extract_json(content)
                logger.debug(f"Extracted JSON from response (try {retry_count+1}): {json_obj}")
                if json_obj:
                    try:
                        jsonschema.validate(json_obj, function_spec.json_schema)
                        
                        if any(_is_empty_value(v) for v in json_obj.values()):
                            logger.warning(f"LLM returned one or more empty values in JSON object: {json_obj}. Retrying...")
                            continue
                        
                        self.history.append({ "role": "assistant", "content": content })
                        logger.debug(f"LLM fallback function call successful, returning: {json_obj}")
                        return json_obj
                    except jsonschema.ValidationError as e:
                        logger.warning(f"JSON schema validation failed for non-native call: {e}. Retrying...")
                        # Fall through to retry
            else:
                self.history.append({ "role": "assistant", "content": content })
                logger.debug(f"LLM text response successful, returning: {content}")
                return content

            logger.warning(f"Failed to parse or validate response, retrying...")

        raise InvalidResponseError("Failed to get a valid response after multiple retries.")
