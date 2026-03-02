"""Shared prompt templates for cache-mode=source (approach B).

When cache_mode=="source", the function source is placed in the system message
(cached across passes for the same function) and each pass sends only its
task-specific instructions as the user message.
"""

FUNCTION_CONTEXT_SYSTEM = """\
You are analyzing the following C/C++ function.

## Function

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

Analyze this function according to the task instructions provided in the user message.\
"""
