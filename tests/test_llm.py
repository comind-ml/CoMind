import pytest
from unittest.mock import MagicMock, patch, call
import json
import jsonschema

from pydantic import SecretStr
from comind.llm.llm import LLM, FunctionSpec, InvalidResponseError
from comind.core.config import LLMConfig

@pytest.fixture
def llm_config():
    """Fixture for LLMConfig."""
    return LLMConfig(
        model_name="o4-mini-2025-04-16",
        api_key=SecretStr("test"),
        max_tokens=1e6,
        temperature=0.7,
        timeout=60,
        max_retries=2,
        keep_history=True,
        native_function_calling=False
    )

@pytest.fixture
def function_spec():
    """Fixture for a sample FunctionSpec."""
    return FunctionSpec(
        name="get_weather",
        description="Get the current weather in a given location",
        json_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    )

def test_function_spec_init_valid():
    """Test FunctionSpec initialization with a valid schema."""
    try:
        FunctionSpec(
            name="test_func",
            description="A test function",
            json_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        )
    except jsonschema.ValidationError:
        pytest.fail("FunctionSpec raised ValidationError unexpectedly with a valid schema.")

def test_function_spec_init_invalid():
    """Test FunctionSpec initialization with an invalid schema."""
    with pytest.raises(jsonschema.SchemaError):
        FunctionSpec(
            name="test_func",
            description="A test function",
            json_schema={
                "type": "object",
                "properties": {
                    "param1": {"type": "invalid_type"}
                }
            }
        )

def test_function_spec_to_openai_tool_dict(function_spec):
    """Test the to_openai_tool_dict method."""
    tool_dict = function_spec.to_openai_tool_dict()
    assert tool_dict == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": function_spec.json_schema
        }
    }

def test_llm_init(llm_config):
    """Test LLM initialization."""
    llm = LLM(llm_config)
    assert llm.config.model_name == "o4-mini-2025-04-16"
    assert llm.use_native_function_calling is True

def test_llm_init_with_kwargs(llm_config):
    """Test LLM initialization with overridden kwargs."""
    llm = LLM(llm_config, model_name="gpt-4o", temperature=0.9)
    assert llm.config.model_name == "gpt-4o"
    assert llm.config.temperature == 0.9
    assert llm.use_native_function_calling is True

def test_llm_validate_json(llm_config, function_spec):
    """Test the validate_json method."""
    llm = LLM(llm_config)
    valid_json = {"location": "Boston, MA", "unit": "celsius"}
    invalid_json = {"location": "Boston, MA", "unit": "kelvin"}
    assert llm.validate_json(valid_json, function_spec.json_schema)
    assert not llm.validate_json(invalid_json, function_spec.json_schema)

@patch('comind.llm.llm.completion')
def test_chat_simple(_, llm_config):
    """Test a simple chat call without function calling."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello, world!"

    llm = LLM(llm_config)
    llm.completion.return_value = mock_response
    response = llm.chat("Hi")

    assert response == "Hello, world!"
    assert len(llm.history) == 2
    assert llm.history[0] == {"role": "user", "content": "Hi"}
    assert llm.history[1] == {"role": "assistant", "content": "Hello, world!"}
    llm.completion.assert_called_once()

@patch('comind.llm.llm.completion')
def test_chat_no_history(_, llm_config):
    """Test chat with keep_history set to False."""
    llm_config.keep_history = False
    
    llm = LLM(llm_config)
    
    # Mock for the first call
    mock_response1 = MagicMock()
    mock_response1.choices[0].message.content = "Response 1"
    llm.completion.return_value = mock_response1
    
    llm.chat("First message")
    # History is populated during the call, but will be cleared on the next one
    assert len(llm.history) == 2 
    
    # Mock for the second call
    mock_response2 = MagicMock()
    mock_response2.choices[0].message.content = "Response 2"
    llm.completion.return_value = mock_response2

    llm.chat("Second message")
    # History should be cleared at the start of the second call and then repopulated
    assert len(llm.history) == 2
    assert llm.history[0]['content'] == "Second message"

@patch('comind.llm.llm.completion')
def test_chat_native_function_call_success(_, llm_config, function_spec):
    """Test a successful native function call."""
    llm_config.model_name = "gpt-4o" # A model that supports native function calling
    llm = LLM(llm_config)

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Tokyo", "unit": "celsius"}'
    
    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message = mock_message
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    llm.completion.return_value = mock_response

    result = llm.chat("What's the weather in Tokyo?", function_spec=function_spec)

    assert result == {"location": "Tokyo", "unit": "celsius"}
    llm.completion.assert_called_once()
    assert len(llm.history) == 2 # user message + assistant tool call message

@patch('comind.llm.llm.completion')
def test_chat_native_function_call_retry_and_fail(_, llm_config, function_spec):
    """Test native function call that fails after retries."""
    llm_config.model_name = "gpt-4o"
    llm = LLM(llm_config)

    # Simulate responses that will fail validation
    mock_invalid_choice = MagicMock()
    mock_invalid_choice.finish_reason = "stop"
    mock_invalid_response = MagicMock()
    mock_invalid_response.choices = [mock_invalid_choice]
    
    llm.completion.return_value = mock_invalid_response

    with pytest.raises(InvalidResponseError):
        llm.chat("Weather in Paris?", function_spec=function_spec)

    assert llm.completion.call_count == llm_config.max_retries

@patch('comind.llm.llm.completion')
def test_chat_fallback_function_call_success(_, llm_config, function_spec):
    """Test a successful fallback (JSON) function call."""
    llm = LLM(llm_config) # Non-native model by default in this config
    
    response_content = """
    Some text...
    ```json
    {
        "location": "London",
        "unit": "celsius"
    }
    ```
    """
    mock_response = MagicMock()
    mock_response.choices[0].message.content = response_content
    llm.completion.return_value = mock_response

    result = llm.chat("Weather in London?", function_spec=function_spec)
    
    assert result == {"location": "London", "unit": "celsius"}
    assert llm.completion.call_count == 1
    user_message_with_prompt = llm.completion.call_args.kwargs['messages'][-1]['content']
    assert function_spec.name in user_message_with_prompt
    assert "must respond using a function call" in user_message_with_prompt

@patch('comind.llm.llm.completion')
def test_chat_fallback_function_call_retry_and_fail(_, llm_config, function_spec):
    """Test fallback function call that fails validation after retries."""
    llm = LLM(llm_config)

    # Simulate responses with invalid JSON
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is not JSON."
    llm.completion.return_value = mock_response
    
    with pytest.raises(InvalidResponseError):
        llm.chat("Weather in Berlin?", function_spec=function_spec)

    assert llm.completion.call_count == llm_config.max_retries

@patch('comind.llm.llm.completion')
def test_chat_native_function_call_empty_value_retry(_, llm_config, function_spec):
    """Test that native function call retries if an argument value is empty."""
    llm_config.model_name = "gpt-4o"
    llm_config.native_function_calling = True # Ensure we test the correct path
    llm = LLM(llm_config)

    # First response: empty value
    mock_tool_call_empty = MagicMock()
    mock_tool_call_empty.function.arguments = '{"location": "", "unit": "celsius"}'
    mock_message_empty = MagicMock(tool_calls=[mock_tool_call_empty], content=None)
    mock_choice_empty = MagicMock(finish_reason="tool_calls", message=mock_message_empty)
    mock_response_empty = MagicMock(choices=[mock_choice_empty])

    # Second response: valid
    mock_tool_call_valid = MagicMock()
    mock_tool_call_valid.function.arguments = '{"location": "Sydney", "unit": "celsius"}'
    mock_message_valid = MagicMock(tool_calls=[mock_tool_call_valid], content=None)
    mock_choice_valid = MagicMock(finish_reason="tool_calls", message=mock_message_valid)
    mock_response_valid = MagicMock(choices=[mock_choice_valid])
    
    llm.completion.side_effect = [mock_response_empty, mock_response_valid]
    
    result = llm.chat("Weather in Sydney?", function_spec=function_spec)
    assert result == {"location": "Sydney", "unit": "celsius"}
    assert llm.completion.call_count == 2

@patch('comind.llm.llm.completion')
def test_chat_fallback_function_call_empty_value_retry(_, llm_config, function_spec):
    """Test that fallback function call retries if a JSON value is empty."""
    llm = LLM(llm_config)

    # First response: empty value
    response_content_empty = '```json\n{"location": "", "unit": "fahrenheit"}\n```'
    mock_response_empty = MagicMock()
    mock_response_empty.choices[0].message.content = response_content_empty
    
    # Second response: valid
    response_content_valid = '```json\n{"location": "New York", "unit": "fahrenheit"}\n```'
    mock_response_valid = MagicMock()
    mock_response_valid.choices[0].message.content = response_content_valid

    llm.completion.side_effect = [mock_response_empty, mock_response_valid]

    result = llm.chat("Weather in New York?", function_spec=function_spec)
    assert result == {"location": "New York", "unit": "fahrenheit"}
    assert llm.completion.call_count == 2 