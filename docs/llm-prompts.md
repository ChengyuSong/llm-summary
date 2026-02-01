# LLM Prompts

This document describes the prompts used to query LLMs for analysis.

## Allocation Summary Prompt

Used to generate memory allocation summaries for functions.

**Location:** `prompts/allocation.txt`

### Structure

```
1. Function source code
2. Function metadata (name, signature, file)
3. Callee summaries (if available)
4. Task description
5. Expected output format
```

### Key Elements

#### Context Provided
- Complete function source code
- Function signature for type information
- Summaries of called functions (compositional analysis)

#### What We Ask For
1. **Allocations**: Memory allocations in the function
   - Type: heap, stack, static
   - Source: allocating function (malloc, calloc, new, etc.)
   - Size expression: how size is computed
   - Size parameters: which parameters affect size
   - Returned: is the allocation returned?
   - Stored to: is it stored to a field/global?
   - May be null: can the allocation fail?

2. **Parameters**: Role of each parameter
   - Role description (size_indicator, buffer, count, etc.)
   - Whether it affects allocation size

3. **Description**: One-sentence summary

### Example Prompt

```
You are analyzing C/C++ code to generate memory allocation summaries.

## Function to Analyze

```c
char* create_buffer(size_t n) {
    char* buf = malloc(n + 1);
    if (!buf) return NULL;
    buf[n] = '\0';
    return buf;
}
```

Function: `create_buffer`
Signature: `char*(size_t)`
File: src/buffer.c

## Callee Summaries

- `malloc`: Allocates size bytes of uninitialized heap memory.

## Task

Generate a memory allocation summary for this function...
```

### Example Response

```json
{
  "function": "create_buffer",
  "allocations": [
    {
      "type": "heap",
      "source": "malloc",
      "size_expr": "n + 1",
      "size_params": ["n"],
      "returned": true,
      "stored_to": null,
      "may_be_null": true
    }
  ],
  "parameters": {
    "n": {
      "role": "size_indicator",
      "used_in_allocation": true
    }
  },
  "description": "Allocates n+1 bytes of heap memory for a null-terminated buffer."
}
```

## Indirect Call Resolution Prompt

Used to determine likely targets for function pointer calls.

**Location:** `prompts/indirect_call.txt`

### Structure

```
1. Indirect call site context
2. Call expression and expected signature
3. Candidate functions (address-taken with matching signature)
4. Address flow information
5. Task: determine likely targets with confidence
```

### Example Prompt

```
You are analyzing indirect function calls in C/C++ code.

## Indirect Call Site

The following indirect call appears in function `process_events`:

```c
// line 142 in events.c
handler->on_complete(ctx, result);
```

The indirect call expression is: `handler->on_complete`
Expected function signature: `void (*)(context_t*, int)`
Location: events.c:142

## Candidate Functions

1. `success_handler` (signature: void(context_t*, int))
   File: handlers.c
   ```c
   void success_handler(context_t* ctx, int result) {
   ```

2. `error_handler` (signature: void(context_t*, int))
   ...

## Address Flow Information

Function `success_handler` address flows to:
  - field:on_complete at handlers.c:45
  - param:register_handler[1] at events.c:98

...

## Task

Determine which functions could realistically be called...
```

### Example Response

```json
{
  "targets": [
    {
      "function": "success_handler",
      "confidence": "high",
      "reasoning": "Assigned to on_complete field in register_handler"
    },
    {
      "function": "error_handler",
      "confidence": "medium",
      "reasoning": "Also assigned to handler callbacks in error paths"
    }
  ]
}
```

## Standard Library Prompt

Used to generate summaries from man pages or documentation.

**Location:** `prompts/stdlib.txt`

### Purpose

Expand stdlib coverage by parsing documentation:
- Man pages
- Header file comments
- Online documentation

### Example

```
Analyze this man page and identify functions that allocate memory.

## Man Page Content

```
GETLINE(3)

NAME
       getline, getdelim - delimited string input

SYNOPSIS
       ssize_t getline(char **lineptr, size_t *n, FILE *stream);

DESCRIPTION
       getline() reads an entire line from stream, storing the address
       of the buffer containing the text into *lineptr. The buffer is
       null-terminated and includes the newline character, if one was
       found.

       If *lineptr is NULL, then getline() will allocate a buffer for
       storing the line, which should be freed by the user program.
...
```

For each function that allocates memory, provide a summary...
```

## Prompt Engineering Guidelines

### Best Practices Used

1. **Clear structure**: Use markdown headers to organize information
2. **Examples in code blocks**: Show exact format expected
3. **JSON output**: Structured, parseable responses
4. **Explicit field descriptions**: No ambiguity in what's expected
5. **Callee context**: Provide summaries of called functions

### Handling Edge Cases

1. **No allocations**: Return empty allocations array
2. **Unknown size**: Use `null` for size_expr
3. **Conditional allocation**: Set may_be_null appropriately
4. **Wrapper functions**: Trace through to underlying allocator

### Response Parsing

The summarizer extracts JSON from responses using:
1. Look for ```json ... ``` blocks
2. Fall back to finding raw JSON objects
3. Handle parse errors gracefully with empty summary

```python
def _parse_response(self, response: str) -> AllocationSummary:
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Fall back to raw JSON
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        ...
```
