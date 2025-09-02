import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agno.exceptions import ModelProviderError, ModelRateLimitError
from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse
from agno.utils.log import log_debug, log_error, log_info, log_warning
from agno.utils.models.claude import format_messages

try:
    from google import genai
    from google.genai import Client as VertexAIClient
    from google.genai.errors import ClientError, ServerError
    from google.genai.types import (
        Content,
        GenerateContentConfig,
        GenerateContentResponse,
        Part,
    )
except ImportError:
    raise ImportError("`google-genai` not installed. Please install it using `pip install google-genai`")


@dataclass
class ClaudeOnVertexAI(Model):
    """
    Claude model available through Google Cloud Vertex AI.
    
    This provides access to Anthropic's Claude models through Google Cloud's Vertex AI platform.
    
    Authentication:
    - You will need Google Cloud credentials to use the Vertex AI API
    - Run `gcloud auth application-default login` to set credentials
    - Set your `project_id` (or set `GOOGLE_CLOUD_PROJECT` environment variable)
    - Set `location` (optional, defaults to "us-central1")
    
    Usage:
        from agno.models.anthropic import ClaudeOnVertexAI
        
        model = ClaudeOnVertexAI(
            id="claude-3-5-sonnet@20241022",
            project_id="your-project-id",
            location="us-central1"
        )
    """

    id: str = "claude-3-5-sonnet@20241022"
    name: str = "Claude on Vertex AI"
    provider: str = "Anthropic (Vertex AI)"

    # Request parameters (Claude-specific)
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    request_params: Optional[Dict[str, Any]] = None

    # Vertex AI parameters
    project_id: Optional[str] = None
    location: Optional[str] = "us-central1"
    client_params: Optional[Dict[str, Any]] = None

    # Client
    client: Optional[VertexAIClient] = None

    def get_client(self) -> VertexAIClient:
        """
        Returns an instance of the Vertex AI client configured for Claude models.
        
        Returns:
            VertexAIClient: The Vertex AI client.
        """
        if self.client:
            return self.client

        client_params: Dict[str, Any] = {
            "vertexai": True,
        }

        # Set project and location
        client_params["project"] = self.project_id or getenv("GOOGLE_CLOUD_PROJECT")
        client_params["location"] = self.location or getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not client_params["project"]:
            log_error("GOOGLE_CLOUD_PROJECT not set. Please set the project_id parameter or GOOGLE_CLOUD_PROJECT environment variable.")

        log_info(f"Using Vertex AI for Claude model: {self.id}")

        # Add additional client parameters
        if self.client_params:
            client_params.update(self.client_params)

        # Filter out None values
        client_params = {k: v for k, v in client_params.items() if v is not None}

        self.client = genai.Client(**client_params)
        return self.client

    def get_request_params(
        self,
        system_message: Optional[str] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate keyword arguments for Vertex AI Claude API requests.
        """
        config_params: Dict[str, Any] = {}

        # Claude-specific parameters
        if self.max_tokens:
            config_params["max_output_tokens"] = self.max_tokens
        if self.temperature:
            config_params["temperature"] = self.temperature
        if self.top_p:
            config_params["top_p"] = self.top_p
        if self.top_k:
            config_params["top_k"] = self.top_k
        if self.stop_sequences:
            config_params["stop_sequences"] = self.stop_sequences

        # System instruction
        if system_message:
            config_params["system_instruction"] = system_message

        # Structured output support
        if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            config_params["response_mime_type"] = "application/json"
            # Note: Full schema support would require additional implementation

        # Additional request parameters
        if self.request_params:
            config_params.update(self.request_params)

        return config_params

    def _format_messages_for_vertex_ai(self, messages: List[Message]) -> List[Content]:
        """
        Convert agno Messages to Vertex AI Content format.
        """
        contents: List[Content] = []
        
        for message in messages:
            if message.role in ["system"]:
                # System messages are handled separately in get_request_params
                continue
            
            # Map roles for Vertex AI
            role = "user" if message.role in ["user", "tool"] else "model"
            
            if isinstance(message.content, str):
                parts = [Part(text=message.content)]
            elif isinstance(message.content, list):
                parts = []
                for content_item in message.content:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "text":
                            parts.append(Part(text=content_item["text"]))
                        elif content_item.get("type") == "tool_result":
                            # Handle tool results
                            parts.append(Part(text=str(content_item.get("content", ""))))
                    else:
                        parts.append(Part(text=str(content_item)))
            else:
                parts = [Part(text=str(message.content))]
            
            contents.append(Content(role=role, parts=parts))
        
        return contents

    def _format_tools_for_vertex_ai(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Transform function definitions into Vertex AI format.
        """
        if not tools:
            return None

        formatted_tools: List[Dict[str, Any]] = []
        for tool_def in tools:
            if tool_def.get("type", "") != "function":
                continue

            func_def = tool_def.get("function", {})
            
            # Convert to Vertex AI function calling format
            vertex_tool = {
                "function_declarations": [{
                    "name": func_def.get("name"),
                    "description": func_def.get("description"),
                    "parameters": func_def.get("parameters", {})
                }]
            }
            formatted_tools.append(vertex_tool)

        return formatted_tools if formatted_tools else None

    def invoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> GenerateContentResponse:
        """
        Send a request to Claude via Vertex AI to generate a response.
        """
        try:
            # Format messages for Vertex AI
            chat_messages, system_message = format_messages(messages)
            vertex_contents = self._format_messages_for_vertex_ai(messages)
            
            # Get request parameters
            config_params = self.get_request_params(system_message, response_format, tools)
            
            # Create generation config
            config = GenerateContentConfig(**config_params)
            
            # Format tools if provided
            vertex_tools = self._format_tools_for_vertex_ai(tools)
            
            log_debug(f"Calling {self.provider} with model: {self.id}")
            
            return self.get_client().models.generate_content(
                model=self.id,
                contents=vertex_contents,
                config=config,
                tools=vertex_tools,
            )
            
        except ClientError as e:
            log_error(f"Vertex AI client error: {str(e)}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except ServerError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                log_warning(f"Rate limit exceeded: {str(e)}")
                raise ModelRateLimitError(message=str(e), model_name=self.name, model_id=self.id) from e
            else:
                log_error(f"Vertex AI server error: {str(e)}")
                raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except Exception as e:
            log_error(f"Unexpected error calling Vertex AI Claude: {str(e)}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    def invoke_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Any:
        """
        Stream a response from Claude via Vertex AI.
        """
        try:
            # Format messages for Vertex AI
            chat_messages, system_message = format_messages(messages)
            vertex_contents = self._format_messages_for_vertex_ai(messages)
            
            # Get request parameters
            config_params = self.get_request_params(system_message, response_format, tools)
            
            # Create generation config
            config = GenerateContentConfig(**config_params)
            
            # Format tools if provided
            vertex_tools = self._format_tools_for_vertex_ai(tools)
            
            log_debug(f"Streaming from {self.provider} with model: {self.id}")
            
            return self.get_client().models.generate_content_stream(
                model=self.id,
                contents=vertex_contents,
                config=config,
                tools=vertex_tools,
            )
            
        except ClientError as e:
            log_error(f"Vertex AI client error: {str(e)}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except ServerError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                log_warning(f"Rate limit exceeded: {str(e)}")
                raise ModelRateLimitError(message=str(e), model_name=self.name, model_id=self.id) from e
            else:
                log_error(f"Vertex AI server error: {str(e)}")
                raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e
        except Exception as e:
            log_error(f"Unexpected error streaming from Vertex AI Claude: {str(e)}")
            raise ModelProviderError(message=str(e), model_name=self.name, model_id=self.id) from e

    async def ainvoke(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> GenerateContentResponse:
        """
        Send an asynchronous request to Claude via Vertex AI.
        """
        # Note: For now, we'll use the synchronous version
        # In a full implementation, we'd need an async Vertex AI client
        return self.invoke(messages, response_format, tools, tool_choice)

    async def ainvoke_stream(
        self,
        messages: List[Message],
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> AsyncIterator[Any]:
        """
        Stream an asynchronous response from Claude via Vertex AI.
        """
        # Note: For now, we'll use the synchronous version wrapped
        # In a full implementation, we'd need an async Vertex AI client
        stream = self.invoke_stream(messages, response_format, tools, tool_choice)
        for chunk in stream:
            yield chunk

    def parse_provider_response(self, response: GenerateContentResponse, **kwargs) -> ModelResponse:
        """
        Parse the Vertex AI Claude response into a ModelResponse.
        
        Args:
            response: Raw response from Vertex AI Claude
            
        Returns:
            ModelResponse: Parsed response data
        """
        model_response = ModelResponse()
        
        # Set role
        model_response.role = "assistant"
        
        # Extract content from candidates
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            if candidate.content and candidate.content.parts:
                content_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content_parts.append(part.text)
                    elif hasattr(part, 'function_call') and part.function_call:
                        # Handle function calls
                        function_call = part.function_call
                        tool_call = {
                            "id": f"call_{hash(str(function_call))}",  # Generate an ID
                            "type": "function",
                            "function": {
                                "name": function_call.name,
                                "arguments": json.dumps(dict(function_call.args))
                            }
                        }
                        model_response.tool_calls.append(tool_call)
                
                if content_parts:
                    model_response.content = "".join(content_parts)
        
        # Extract usage information
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = response.usage_metadata
            model_response.response_usage = {
                "input_tokens": getattr(usage_metadata, 'prompt_token_count', 0),
                "output_tokens": getattr(usage_metadata, 'candidates_token_count', 0),
                "total_tokens": getattr(usage_metadata, 'total_token_count', 0),
            }
        
        return model_response

    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        """
        Parse streaming response chunks from Vertex AI Claude.
        
        Args:
            response: Raw response chunk from Vertex AI Claude
            
        Returns:
            ModelResponse: Parsed response data
        """
        model_response = ModelResponse()
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            if candidate.content and candidate.content.parts:
                content_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content_parts.append(part.text)
                
                if content_parts:
                    model_response.content = "".join(content_parts)
        
        return model_response

    def format_function_call_results(self, messages: List[Message], function_call_results: List[Message]) -> None:
        """
        Handle the results of function calls for Vertex AI Claude.
        
        Args:
            messages (List[Message]): The list of conversation messages.
            function_call_results (List[Message]): The results of the function calls.
        """
        if len(function_call_results) > 0:
            for fc_message in function_call_results:
                # Format function results as user messages with tool content
                result_content = [{
                    "type": "tool_result", 
                    "tool_use_id": fc_message.tool_call_id,
                    "content": str(fc_message.content)
                }]
                messages.append(Message(role="user", content=result_content))

    def get_system_message_for_model(self, tools: Optional[List[Any]] = None) -> Optional[str]:
        """
        Get system message for Claude on Vertex AI.
        """
        if tools is not None and len(tools) > 0:
            return "You are a helpful assistant that can use tools to answer questions. Use the available tools when appropriate."
        return None