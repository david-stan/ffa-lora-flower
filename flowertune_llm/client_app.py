"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
# from flwr.client.mod import fixedclipping_mod

from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
    format_gsm8k_prompt,
)
from flowertune_llm.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)



import dp_transformers
from dp_transformers.dp_utils import OpacusDPTrainer


# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        privacy_cfg: DictConfig,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        self.privacy_arguments = dp_transformers.PrivacyArguments(
            target_epsilon=6,
            target_delta=1e-5,
            per_sample_max_grad_norm=2.0,
            disable_dp=False
        )
        

        # instantiate model
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # Construct trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     args=self.training_argumnets,
        #     max_seq_length=self.train_cfg.seq_length,
        #     train_dataset=self.trainset,
        #     formatting_func=self.formatting_prompts_func,
        #     data_collator=self.data_collator,
        # )
        dataset = self.trainset.train_test_split(0.02)
        dataset = dataset.map(format_gsm8k_prompt, batched=False, desc="formatting GSM8K prompts")
        dataset = dataset.map(
            lambda batch: self.tokenizer(batch['text'], padding="max_length", truncation=True, max_length=self.train_cfg.seq_length),
            batched=True, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
        )

        trainer = OpacusDPTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_argumnets,
            # max_seq_length=self.train_cfg.seq_length,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            # formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            privacy_args=self.privacy_arguments,
        )

        # Do local training
        if hasattr(trainer.model._module, "config"):
            # The following is for GradSampleModule wrapping
            ignore_keys = getattr(trainer.model._module.config, "keys_to_ignore_at_inference", [])
        elif hasattr(trainer.model._module.module, "config"):
            # The following is for GradSampleModule and DPDDP wrapping
            ignore_keys = getattr(trainer.model._module.module.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

        try:
            # A workaround to avoid the following error:
            # AttributeError: 'GradSampleModule' object has no attribute 'gradient_checkpointing_enable'
            # inside Trainer _inner_training_loop. Already done by prepare_model_for_kbit_training
            trainer.args.gradient_checkpointing = False
            result = trainer.train(ignore_keys_for_eval=ignore_keys)
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })
            print(eps_prv)
            print(eps_rdp)

        # trainer = Trainer(
        #     args=self.training_argumnets,
        #     model=self.model,
        #     train_dataset=dataset['train'],
        #     eval_dataset=dataset['test'],
        #     data_collator=self.data_collator,
        # )

        print("######################3##########")
        print(trainer.args.optim)
        print(trainer.args.learning_rate)
        print(trainer.label_names)
        print("######################3##########")

        # result = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": result.training_loss},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_trainset,
        tokenizer,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        cfg.privacy,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)