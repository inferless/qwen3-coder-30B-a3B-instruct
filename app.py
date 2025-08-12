import torch
import os, inferless
from typing import Optional
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Write a quick sort algorithm.")
    system_prompt: Optional[str] = "You are a coding expert"
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 1.0
    do_sample: Optional[bool] = True

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        # Dump environment info to the console

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        # Prepare messages
        messages = [{"role": "system", "content": inputs.system_prompt},
                    {"role": "user", "content": inputs.prompt}
                   ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate text
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=inputs.max_new_tokens,
                temperature=inputs.temperature,
                top_p=inputs.top_p,
                do_sample=inputs.do_sample
            )
        
        # Extract only the generated tokens (excluding input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Decode the generated content
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return ResponseObjects(generated_text=content)

    def finalize(self):
        self.model = None
