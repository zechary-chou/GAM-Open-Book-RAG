import torch
import json

from typing import List, Union, Dict, Any, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

from gam.generator.base import AbsGenerator
from config import VLLMGeneratorConfig


class VLLMGenerator(AbsGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.generate_model_path = config.get("generate_model_path", "Qwen/Qwen2.5-7B-Instruct")
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.8)
        self.tensor_parallel_size = config.get("tensor_parallel_size")
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 1.0)
        self.max_tokens = config.get("max_tokens", 300)
        self.stop = config.get("stop")
        self.repetition_penalty = config.get("repetition_penalty", 1.1)
        self.lora_path = config.get("lora_path")
        self.n = config.get("n", 1)
        self.system_prompt = config.get("system_prompt")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.generate_model_path, trust_remote_code=True)
        self.llm = LLM(model=self.generate_model_path, gpu_memory_utilization=self.gpu_memory_utilization,
                       trust_remote_code=True,
                       tensor_parallel_size=torch.cuda.device_count() if self.tensor_parallel_size is None else self.tensor_parallel_size,
                       enable_lora=True, max_lora_rank=64)
        self.model_name = self.generate_model_path.lower()

    def generate_single(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        极简 Chat 调用（VLLM）
        - 二选一：prompt 文本 或 messages 列表
        - 若传 schema：进行结构化输出（VLLM 需要特殊处理）
        返回：
          {"text": str, "json": dict|None, "response": dict}
        """
        if (prompt is None) and (not messages):
            raise ValueError("Either prompt or messages is required.")
        if (prompt is not None) and messages:
            raise ValueError("Pass either prompt or messages, not both.")

        # 构造 messages
        if messages is None:
            messages = [{"role": "user", "content": prompt}]  # type: ignore[arg-type]
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages = [{"role": "system", "content": self.system_prompt}] + messages

        sampling_params = SamplingParams(
            temperature=self.temperature, 
            top_p=self.top_p, 
            max_tokens=self.max_tokens,
            stop=self.stop or [], 
            n=self.n,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=[128001, 128009] if 'llama' in self.model_name else None
        )
        lora_request = None
        if self.lora_path is not None:
            lora_request = LoRARequest("lora_adapter", 1, self.lora_path)

        formatted_prompt = self.tokenizer.decode(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)[1:]
        )
        outputs = self.llm.generate([formatted_prompt], sampling_params, lora_request=lora_request)

        # 获取生成的文本
        if self.n > 1:
            text = outputs[0].outputs[0].text.strip()
            texts = [outputs[0].outputs[i].text.strip() for i in range(len(outputs[0].outputs))]
        else:
            text = outputs[0].outputs[0].text.strip()
            texts = [text]

        # 构造返回格式
        out: Dict[str, Any] = {
            "text": text, 
            "json": None, 
            "response": {
                "outputs": [{"text": t} for t in texts],
                "model": self.model_name,
                "sampling_params": sampling_params.__dict__
            }
        }

        # 如果有 schema，尝试解析 JSON
        if schema is not None:
            try:
                out["json"] = json.loads(text)
            except Exception:
                out["json"] = None

        return out

    def generate_batch(
        self,
        prompts: Optional[List[str]] = None,
        messages_list: Optional[List[List[Dict[str, str]]]] = None,
        schema: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成响应
        返回格式: [{"text": str, "json": dict|None, "response": dict}, ...]
        """
        if (prompts is None) and (not messages_list):
            raise ValueError("Either prompts or messages_list is required.")
        if (prompts is not None) and messages_list:
            raise ValueError("Pass either prompts or messages_list, not both.")

        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            # 转换为 messages_list 格式
            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

        sampling_params = SamplingParams(
            temperature=self.temperature, 
            top_p=self.top_p, 
            max_tokens=self.max_tokens,
            stop=self.stop or [], 
            n=self.n,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=[128001, 128009] if 'llama' in self.model_name else None
        )
        lora_request = None
        if self.lora_path is not None:
            lora_request = LoRARequest("lora_adapter", 1, self.lora_path)

        # 为每个 messages 添加 system_prompt
        formatted_messages_list = []
        for messages in messages_list:
            if self.system_prompt and not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": self.system_prompt}] + messages
            formatted_messages_list.append(messages)

        formatted_prompts = [self.tokenizer.decode(
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)[1:]
        ) for messages in formatted_messages_list]
        
        outputs = self.llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        results = []
        for output in outputs:
            if self.n > 1:
                text = output.outputs[0].text.strip()
                texts = [output.outputs[i].text.strip() for i in range(len(output.outputs))]
            else:
                text = output.outputs[0].text.strip()
                texts = [text]

            # 构造返回格式
            out = {
                "text": text, 
                "json": None, 
                "response": {
                    "outputs": [{"text": t} for t in texts],
                    "model": self.model_name,
                    "sampling_params": sampling_params.__dict__
                }
            }

            # 如果有 schema，尝试解析 JSON
            if schema is not None:
                try:
                    out["json"] = json.loads(text)
                except Exception:
                    out["json"] = None

            results.append(out)
        return results
    
    @classmethod
    def from_config(cls, config: VLLMGeneratorConfig) -> "VLLMGenerator":
        """从配置类创建 VLLMGenerator 实例"""
        return cls(config.__dict__)