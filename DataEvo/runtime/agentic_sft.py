"""Install a prefix-stable ChatML template before SFT and checkpoint saving."""
from DataEvo.runtime.common import REPO
from DataEvo.runtime.agentic_protocol import token_spans
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset, convert_nested_value_to_list_recursive


def configure_tokenizer(tokenizer):
    vocabulary = tokenizer.get_vocab()
    if '<|im_start|>' not in vocabulary or '<|im_end|>' not in vocabulary:
        raise ValueError('terminal profile currently requires a ChatML tokenizer (e.g. Qwen)')
    tokenizer.chat_template = (REPO / 'DataEvo/config/terminal_chatml.jinja').read_text()
    return tokenizer


class TerminalSFTDataset(MultiTurnSFTDataset):
    def __init__(self, parquet_files, tokenizer, config=None):
        super().__init__(parquet_files, configure_tokenizer(tokenizer), config)

    def __getitem__(self, index):
        import torch
        messages = convert_nested_value_to_list_recursive(self.messages[index])
        first = next((i for i, message in enumerate(messages) if message['role'] == 'assistant'), None)
        if first is None or first < 1:
            raise ValueError('terminal SFT requires a prompt followed by assistant actions')
        initial, response, mask = token_spans(self.tokenizer, messages, first)
        tokens, mask = initial + response, [0] * len(initial) + mask
        if len(tokens) > self.max_length:
            raise ValueError('terminal SFT trajectory exceeds max_length; filter it or raise the fixed limit')
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        remaining = self.max_length - len(tokens)
        attention = torch.tensor([1] * len(tokens) + [0] * remaining, dtype=torch.long)
        return {'input_ids': torch.tensor(tokens + [pad] * remaining, dtype=torch.long),
                'attention_mask': attention, 'loss_mask': torch.tensor(mask + [0] * remaining, dtype=torch.long),
                'position_ids': torch.arange(self.max_length, dtype=torch.long) * attention}
