from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """LLM基类，定义统一的接口"""
    
    @abstractmethod
    def generate(self, message, **kwargs):
        """生成文本的抽象方法"""
        pass


class HFModel(BaseLLM):
    def __init__(self, model_name, temperature=0.1, top_p=1.0, max_new_tokens=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(self.device)

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

    def generate(self, message, **kwargs):
        # 自动获取模型的第一个参数所在设备
        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer([message], return_tensors="pt").to(model_device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
 

class OpenRouterModel(BaseLLM):
    def __init__(self, model="openai/gpt-5", base_url="https://api2.aigcbest.top/v1", extra_headers=None, extra_body=None, max_retries: int = 2):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key='',
        )
        self.model = model
        self.max_retries = max_retries
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}

    def generate(self, message, **kwargs):
        """
        message: str 或 list，支持纯文本或多模态（如图片）
        kwargs: 额外参数，如 temperature, top_p 等
        """
        # 构造 messages
        if isinstance(message, str):
            messages = [{"role": "user", "content": message}]
        elif isinstance(message, list):
            messages = [{"role": "user", "content": message}]
        else:
            raise ValueError("message 必须为 str 或 list")

        for attempt in range(self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **kwargs
                )
                choice0 = (completion.choices or [None])[0]
                if not choice0 or not getattr(choice0, "message", None):
                    raise RuntimeError(f"Empty choices/message. Raw obj: {completion}")
                return choice0.message.content

            except Exception as e:
                # 打印更有用的信息，方便定位是鉴权/限流/404/502 还是消息结构问题
                err_type = type(e).__name__
                detail = getattr(e, "message", None) or str(e)
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                print(f"[OpenRouter generate] attempt={attempt} error={err_type} status={status} detail={detail}")

                last_err = e
                if attempt < self.max_retries:
                    continue
                raise last_err