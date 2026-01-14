from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


class LoopLM(nn.Module):
    def __init__(self, base_causallm, n_loop, **kwargs):
        super(LoopLM, self).__init__()
        self.base_causallm = base_causallm
        self.n_loop = n_loop

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, **kwargs):
        inputs_embeds = self.base_causallm.get_input_embeddings()(input_ids)

        hidden_states = None
        for _ in range(self.n_loop):
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            inputs_embeds = hidden_states  # update inputs_embeds for the next loop

        logits = self.base_causallm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=torch.tensor([tokens], device=input_ids.device),
                attention_mask=None,
                position_ids=None,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            tokens.append(next_token)

        generated = torch.tensor([tokens], device=input_ids.device)

        if output_embedding:
            embeddings = self.base_causallm.get_input_embeddings()(generated)
            return embeddings
        else:
            return generated
