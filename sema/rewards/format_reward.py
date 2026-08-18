# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re

class NumberedListFormatReward:
    def __init__(self, max_number=None, pattern=r'(\d+\.)\s*(.*?)(?=\n\d+\.|$)', return_dict=True):
        self.max_number = max_number
        self.pattern = pattern
        self.return_dict = return_dict
    
    def reward(self, completion, return_dict=None):
        if return_dict is None:
            return_dict = self.return_dict
        try:
            matched_numbered_list = re.findall(self.pattern, completion[-1]['content'], re.DOTALL)
            numbers = [int(item[0][:-1]) for item in matched_numbered_list]
            parsed_items = [item[1].strip() for item in matched_numbered_list]
            if self.max_number is not None:
                parsed_items = parsed_items[:self.max_number]
            if len(parsed_items) > 0:
                expected_numbers = list(range(1, len(parsed_items) + 1))
                if numbers == expected_numbers:
                    reward = 1
                else:
                    reward = 0
            else:
                reward = 0
        except Exception as e:
            print(f"Error processing completion: {e}")
            reward = 0
            parsed_items = []
        
        if return_dict:
            reward_dict = {
                "reward": reward,
                "parsed_items": parsed_items
            }
            return reward_dict
        else:
            return reward

    def reward_batch(self, completions, return_dict=None):
        if return_dict is None:
            return_dict = self.return_dict
        outputs = []
        for completion in completions:
            output = self.reward(completion, return_dict)
            outputs.append(output)
        return outputs