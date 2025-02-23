import copy
import torch
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

def get_base_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"]
    )

    configuration = RobertaConfig.from_pretrained(
        model_id,
        num_labels=3,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        model_id,
        do_lower_case=False,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        model_id,
        config=configuration,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    return model, tokenizer

def get_lora_model_from_base(model, vanilla_lora=False):
    model_copy = copy.deepcopy(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters count: {total_params:,}")


    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,         # our particular task is sequence classification
        inference_mode=False,               # enable training mode
        r=8,                                # Low-rank dimension
        lora_alpha=8,                       # Alpha scaling factor
        lora_dropout=0.05,                  # dropout for LoRA layers
        target_modules=["query", "value"],
    )

    model_with_lora = get_peft_model(model_copy, lora_config)
    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    print(f"Total trainable parameters with LoRA: {trainable_params:,}")

    if vanilla_lora:
        for name, param in model_with_lora.named_parameters():
            if "classifier" in name:
                param.requires_grad = False
    else:
        # FFA-LoRA modification: freeze all adapter A matrices so that only B matrices are trainable
        for name, param in model_with_lora.named_parameters():
            if "lora_A" in name or "classifier" in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
    print(f"Total trainable parameters with LoRA after freezing: {trainable_params:,}")

    return model_with_lora
