import tiktoken
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class TokenUsage:
    """Token usage tracking data structure"""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str
    agent_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name
        }

class TokenManager:
    """Token usage calculator and limiter for different models"""
    
    # Model token limits (input + output combined)
    MODEL_LIMITS = {
        "gpt-4.1-mini": 1047576,
        "gpt-4o-mini":   128000,
        "gpt-4.1":      1047576,
        "gpt-4o":        128000
    }
    
    # Model encodings
    MODEL_ENCODINGS = {
        "gpt-4.1-mini": "cl100k_base",
        "gpt-4o-mini": "cl100k_base", 
        "gpt-4.1": "cl100k_base",
        "gpt-4o": "cl100k_base",
    }
    
    def __init__(self):
        self.usage_history: List[TokenUsage] = []
        self._encoders = {}
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create tiktoken encoder for the model"""
        encoding_name = self.MODEL_ENCODINGS.get(model, "cl100k_base")
        if encoding_name not in self._encoders:
            self._encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
        return self._encoders[encoding_name]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for specific model"""
        if not text:
            return 0
        
        encoder = self._get_encoder(model)
        return len(encoder.encode(str(text)))
    
    def get_model_limit(self, model: str) -> int:
        """Get token limit for specific model"""
        return self.MODEL_LIMITS.get(model, 8192)  # Default to GPT-4 original limit
    
    def estimate_tokens_for_agent_input(self, 
                                      instructions: str, 
                                      user_input: str, 
                                      model: str,
                                      additional_context: str = "") -> int:
        """Estimate input tokens for agent request"""
        # Combine all input components
        total_input = f"{instructions}\n{user_input}\n{additional_context}"
        return self.count_tokens(total_input, model)
    
    def check_token_limit(self, 
                         estimated_input_tokens: int, 
                         model: str, 
                         safety_margin: float = 0.1) -> Tuple[bool, int, int]:
        """
        Check if estimated tokens exceed model limit with safety margin
        
        Args:
            estimated_input_tokens: Estimated input tokens
            model: Model name
            safety_margin: Safety margin as percentage (0.1 = 10%)
            
        Returns:
            (is_within_limit, available_tokens, model_limit)
        """
        model_limit = self.get_model_limit(model)
        safe_limit = int(model_limit * (1 - safety_margin))
        available_tokens = safe_limit - estimated_input_tokens
        
        return available_tokens > 0, available_tokens, model_limit
    
    def log_usage(self, 
                  model: str, 
                  input_tokens: int, 
                  output_tokens: int, 
                  agent_name: str = ""):
        """Log token usage"""
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name
        )
        self.usage_history.append(usage)
        
        print(f"🔢 Token Usage - {agent_name} ({model}): "
              f"Input: {input_tokens}, Output: {output_tokens}, Total: {usage.total_tokens}")
    
    def get_total_usage(self) -> Dict[str, int]:
        """Get total token usage across all models"""
        total_by_model = {}
        for usage in self.usage_history:
            if usage.model not in total_by_model:
                total_by_model[usage.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0
                }
            
            total_by_model[usage.model]["input_tokens"] += usage.input_tokens
            total_by_model[usage.model]["output_tokens"] += usage.output_tokens
            total_by_model[usage.model]["total_tokens"] += usage.total_tokens
            total_by_model[usage.model]["requests"] += 1
        
        return total_by_model
    
    def get_session_summary(self) -> str:
        """Get formatted summary of token usage for this session"""
        total_usage = self.get_total_usage()
        
        if not total_usage:
            return "📊 本次會話尚未使用任何 token"
        
        summary_lines = ["📊 本次會話 Token 使用統計:"]
        grand_total = 0
        
        for model, usage in total_usage.items():
            grand_total += usage["total_tokens"]
            model_limit = self.get_model_limit(model)
            usage_percent = (usage["total_tokens"] / model_limit) * 100
            
            summary_lines.append(
                f"  • {model}: {usage['total_tokens']:,} tokens "
                f"({usage['requests']} 次請求, 使用 {usage_percent:.1f}% 容量)"
            )
        
        summary_lines.append(f"  • 總計: {grand_total:,} tokens")
        
        return "\n".join(summary_lines)
    
    def save_usage_log(self, filepath: str):
        """Save usage history to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([usage.to_dict() for usage in self.usage_history], 
                     f, ensure_ascii=False, indent=2)
    
    def truncate_text_to_fit(self, 
                           text: str, 
                           model: str, 
                           reserved_tokens: int = 1000) -> str:
        """
        Truncate text to fit within model limits
        
        Args:
            text: Text to potentially truncate
            model: Model name
            reserved_tokens: Tokens to reserve for other parts of prompt
            
        Returns:
            Potentially truncated text
        """
        max_tokens = self.get_model_limit(model) - reserved_tokens
        current_tokens = self.count_tokens(text, model)
        
        if current_tokens <= max_tokens:
            return text
        
        # Binary search to find appropriate truncation point
        encoder = self._get_encoder(model)
        tokens = encoder.encode(text)
        
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoder.decode(truncated_tokens)
            print(f"⚠️  文本已截斷: 原 {current_tokens} tokens → {max_tokens} tokens")
            return truncated_text
        
        return text

# Global token manager instance
token_manager = TokenManager()

def get_token_manager() -> TokenManager:
    """Get the global token manager instance"""
    return token_manager