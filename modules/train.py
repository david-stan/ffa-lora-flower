import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from tqdm import tqdm

from modules.utils import evaluate


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def federated_train(
        model_with_lora,
        client_train_partitions,
        test_dataloader,
        config, 
        vanilla_lora=True,
        logging=True,
    ):
    device = "cuda"

    global_model = copy.deepcopy(model_with_lora).to(device)
    global_model.train()

    num_clients = config['NUM_CLIENTS']

    global_loss_list = []
    global_acc_list = []


    for round in range(1, config['NUM_GLOBAL_ROUNDS'] + 1):
        print(f"Global Round {round}")
        client_params_list = []

        for client_id in range(num_clients):

            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()
            # local dataloader
            local_train_loader = DataLoader(
                client_train_partitions[client_id],
                sampler=RandomSampler(client_train_partitions[client_id]),
                batch_size=config['BATCH_SIZE'],
            )

            learning_rate = 0.2
            if vanilla_lora:
                learning_rate = 0.02
            
            local_optimizer = torch.optim.SGD(local_model.parameters(), lr=learning_rate)
            privacy_engine = PrivacyEngine()

            local_model, local_optimizer, local_train_loader = privacy_engine.make_private_with_epsilon(
                module=local_model,
                optimizer=local_optimizer,
                data_loader=local_train_loader,
                target_delta=config['DELTA'],
                target_epsilon=config['EPSILON'],
                epochs=config['EPOCHS'],  # local training epochs
                max_grad_norm=config['MAX_GRAD_NORM'],
            )

            for _ in range(1, config['EPOCHS']+1):
                local_losses = []
                with BatchMemoryManager(
                    data_loader=local_train_loader,
                    max_physical_batch_size=config['MAX_PHYSICAL_BATCH_SIZE'],
                    optimizer=local_optimizer
                ) as memory_safe_loader:               
                    for step, batch in enumerate(tqdm(memory_safe_loader, desc=f"   Client {client_id} training")):
                        if step < 400:
                            local_optimizer.zero_grad()
                            batch = tuple(t.to(device) for t in batch)
                            inputs = {
                                'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'labels': batch[2],
                            }
                            outputs = local_model(**inputs)
                            loss = outputs[0]
                            loss.backward()
                            local_losses.append(loss.item())
                            local_optimizer.step()
                            
                            if step > 0 and step % config['LOGGING_INTERVAL'] == 0 and logging:
                                train_loss = np.mean(local_losses)
                                eps = privacy_engine.get_epsilon(config['DELTA'])

                                eval_loss, eval_accuracy = evaluate(local_model, test_dataloader)

                                print(
                                    f"      Step: {step} | "
                                    f"Train loss: {train_loss:.3f} | "
                                    f"Eval loss: {eval_loss:.3f} | "
                                    f"Eval accuracy: {eval_accuracy:.3f} | "
                                    f"ɛ: {eps:.2f}"
                                )

            print(f"    Local training loss: {np.mean(local_losses):.3f}")
            # get parameters
            local_params = get_peft_model_state_dict(local_model.base_model)
            client_params_list.append(local_params)

        # aggregate
        aggregated_params = FedAvg(client_params_list)
        # update ~.base_model!!!
        set_peft_model_state_dict(global_model.base_model, aggregated_params)

        # eval global
        global_eval_loss, global_eval_accuracy = evaluate(global_model, test_dataloader)

        global_loss_list.append(global_eval_loss)
        global_acc_list.append(global_eval_accuracy)

        print(
            f"After Global Round {round}: Eval loss: {global_eval_loss:.3f} | "
            f"Eval accuracy: {global_eval_accuracy:.3f}"
        )

    return global_model, global_acc_list, global_loss_list