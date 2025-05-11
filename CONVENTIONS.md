# LLM Coding Instructions for Python 3.12

Follow these instructions when generating or modifying Python code for this project. Your primary goal is to produce clean, readable, and maintainable code that adheres to Python best practices and project-specific standards.

## Core Python Standards (Adhere Strictly)

### 1. PEP 8: Style Guide
Your code MUST conform to PEP 8. Key directives include:
- **Indentation:** Use 4 spaces per level.
- **Line Length:** Max 79 characters for code, 72 for docstrings/comments. (Code will be formatted by `black` which may use 88 chars, but aim for 79 initially).
- **Blank Lines:**
    - Two for top-level function/class definitions.
    - One for method definitions within a class.
    - Use judiciously to separate logical sections.
- **Imports:**
    - One import per line.
    - Order: 1. Standard library, 2. Third-party, 3. Local application/library.
    - Place at the top of the file (after module docstring/comments, before globals).
- **Whitespace:**
    - Avoid extraneous whitespace (e.g., inside parentheses, before commas).
    - Follow standard practices around operators.
- **Naming Conventions:**
    - `snake_case` for functions, methods, and variables.
    - `PascalCase` (or `CapWords`) for classes.
    - `UPPERCASE_WITH_UNDERSCORES` for constants.
    - `self` as the first argument for instance methods.
    - `cls` as the first argument for class methods.
- **Programming Recommendations:**
    - Use `is` or `is not` for `None` comparisons (e.g., `if my_var is None:`).
    - Use `isinstance()` for type checking (e.g., `isinstance(obj, int)`).
    - For sequences (strings, lists, tuples), check for emptiness directly (e.g., `if not my_list:`).

### 2. PEP 257: Docstring Conventions
Your generated code MUST include comprehensive docstrings.
- **Coverage:** All public modules, functions, classes, and methods MUST have docstrings.
- **Format:**
    - First line: Concise summary of the object's purpose.
    - Multi-line: Summary line, then a blank line, then a more detailed explanation.
    - Describe arguments, return values, and any exceptions raised.

### 3. PEP 20: The Zen of Python
Let the principles of PEP 20 guide your code generation. Strive for code that is:
- Beautiful, Explicit, Simple, Readable.
- "There should be one-- and preferably only one --obvious way to do it."
- If the implementation is hard to explain, it's a bad idea.

### 4. Type Hinting (PEP 484 and related)
Your code MUST use type hints for all function signatures (arguments and return types) and variable annotations where appropriate.
- Use modern type hinting syntax (e.g., `list[int]` instead of `typing.List[int]` if Python 3.9+ is assumed, which it is for 3.12).
- Follow PEP 484, PEP 526, PEP 544, PEP 585, PEP 604, and PEP 695.

## Project-Specific Instructions (Adhere Strictly)

### 1. Code Conciseness and Structure:
- **Write concise code.** Avoid unnecessary verbosity.
- **Refactor deeply nested `if/else` blocks.** Employ techniques like:
    - Guard clauses (early exits).
    - Dictionary-based dispatch.
    - Polymorphism.

### 2. File Organization: One Class Per File
- **Each class definition MUST reside in its own Python file.**
- **The filename MUST match the class name in `snake_case`.** For example, a class `MyExampleClass` should be in a file named `my_example_class.py`.

### 3. Exception Handling:
- **Allow general exceptions (`Exception` and its sub-classes like `RuntimeError`, `ValueError`, etc., unless very specific) to propagate up the call stack.** Do not catch them in lower-level functions just to log and re-raise, unless the logging adds critical contextual information not available higher up.
- **Implement specific, meaningful exception handling where it adds value to program flow or error recovery.** For example, catching `FileNotFoundError` to provide a default or `TypeError`/`ValueError` for input validation is appropriate.

### 4. Code Formatting (Black):
- While you should aim for PEP 8 compliance, be aware that the final code formatting will be handled by the `black` auto-formatter.
- **Generate code that is `black`-compatible.** This generally means adhering to `black`'s default line length (often 88 characters, though strive for 79 as per PEP 8 initially) and its other styling choices.

### 5. Comments (Non-Docstring):
- **Write meaningful comments that explain *why* something is done or clarify complex logic.**
- **Do NOT write comments that merely state *what* the code is doing if the code itself is self-explanatory.**
- **Avoid non-documenting comments,** such as "fixed bug," "removed this," "added that," or comments stating the obvious.
- Docstrings (PEP 257) are for API documentation; regular comments are for implementation clarity.