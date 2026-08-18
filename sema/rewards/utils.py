# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import json


def remove_markdown_json(response_msg: str) -> str:
    """
    Checks if the response message is in JSON format and removes Markdown formatting if present.

    Args:
        response_msg (str): The response message to check.

    Returns:
        str: The response message without Markdown formatting if present.
    """

    start_pattern = re.compile(r"^(```json\n|`json\n|```\n|`\n|```json|`json|```|`|json|json\n)")
    match = start_pattern.match(response_msg)
    if match:
        response_msg = response_msg[match.end() :]
    end_pattern = re.compile(r"(\n```|\n`|```|`)$")
    
    match = end_pattern.search(response_msg)
    if match:
        response_msg = response_msg[: match.start()]

    # Validate if the remaining response message is valid JSON. If it's still not valid
    # after removing the markdown notation, try to extract JSON from within the string.
    try:
        json.loads(response_msg)
        return response_msg
    except json.JSONDecodeError:
        json_pattern = re.compile(r"\{.*\}|\[.*\]")
        match = json_pattern.search(response_msg)
        if match:
            response_msg = match.group(0)
        try:
            json.loads(response_msg)
            return response_msg
        except json.JSONDecodeError:
            return "Invalid JSON response: {}".format(response_msg)
