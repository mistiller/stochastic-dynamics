# Coding Conventions

This document outlines the coding conventions and style guidelines for this project, targeting Python 3.12. It combines general Python best practices with project-specific preferences.

## General Python Coding Standards

We adhere to the widely accepted Python Enhancement Proposals (PEPs) for styling and conventions.

### PEP 8: Style Guide for Python Code
PEP 8 is the primary style guide for Python code. It covers aspects like code layout, naming conventions, comments, and more, aiming to improve the readability and consistency of Python code.
- **Reference:** [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- **Key Aspects:**
    - **Indentation:** Use 4 spaces per indentation level.
    - **Line Length:** Limit all lines to a maximum of 79 characters. For flowing long blocks of text with fewer structural restrictions (docstrings or comments), the line length should be limited to 72 characters. (Note: `black` formatter defaults to 88 characters).
    - **Blank Lines:** Use blank lines to separate functions and classes, and larger blocks of code inside functions. Two blank lines for top-level function and class definitions. One blank line for method definitions inside a class.
    - **Imports:** Imports should usually be on separate lines. E.g., `import os` and `import sys`, not `import os, sys`. Imports are always put at the top of the file, just after any module comments and docstrings, and before module globals and constants. Imports should be grouped in the following order:
        1. Standard library imports.
        2. Related third-party imports.
        3. Local application/library specific imports.
    - **Whitespace in Expressions and Statements:** Avoid extraneous whitespace in the following situations:
        - Immediately inside parentheses, brackets or braces.
        - Immediately before a comma, semicolon, or colon.
        - However, in a slice, the colon acts like a binary operator, and should have equal amounts on either side (treating it as the operator with the lowest priority). In an extended slice, both colons must have the same amount of spacing applied.
        - Immediately before the open parenthesis that starts the argument list of a function call.
        - Immediately before the open parenthesis that starts an indexing or slicing.
        - More than one space around an assignment (or other) operator to align it with another.
    - **Naming Conventions:**
        - `snake_case` for functions and variable names.
        - `PascalCase` (or `CapWords`) for class names.
        - `UPPERCASE_WITH_UNDERSCORES` for constants.
        - Function and method arguments: `self` for the first argument to an instance method, and `cls` for the first argument to a class method.
    - **Comments:** Write comments that are clear, concise, and up to date.
        - Block comments generally apply to some (or all) code that follows them, and are indented to the same level as that code. Each line of a block comment starts with a `#` and a single space.
        - An inline comment is a comment on the same line as a statement. Inline comments should be used sparingly. An inline comment is separated by at least two spaces from the statement. They should start with a # and a single space.
    - **Programming Recommendations:**
        - Comparisons to singletons like `None` should always be done with `is` or `is not`, never the equality operators.
        - Use `isinstance()` for type checking of objects, e.g. `isinstance(obj, int)`.
        - For sequences, (strings, lists, tuples), use the fact that empty sequences are false. `if not my_list:` is preferred over `if len(my_list) == 0:`.

### PEP 257: Docstring Conventions
PEP 257 provides conventions for docstrings, which are essential for documenting what your code does.
- **Reference:** [PEP 257 -- Docstring Conventions](https://peps.python.org/pep-0257/)
- **Key Aspects:**
    - Write docstrings for all public modules, functions, classes, and methods.
    - The first line of a docstring should be a short, concise summary of the object's purpose.
    - For multi-line docstrings, the summary line is followed by a blank line, then a more detailed explanation.

### PEP 20: The Zen of Python
PEP 20, by Tim Peters, is a collection of 19 guiding principles for writing computer programs that influence the design of the Python language. You can view them by typing `import this` into a Python interpreter.
- **Reference:** [PEP 20 -- The Zen of Python](https://peps.python.org/pep-0020/)
- **Key Aphorisms Include:**
    - Beautiful is better than ugly.
    - Explicit is better than implicit.
    - Simple is better than complex.
    - Complex is better than complicated.
    - Readability counts.
    - There should be one-- and preferably only one --obvious way to do it.
    - If the implementation is hard to explain, it's a bad idea.
    - If the implementation is easy to explain, it may be a good idea.

### Type Hinting (PEP 484 and related)
We use type hints to improve code clarity and for static analysis.
- **Reference:** [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
- Subsequent PEPs have expanded on type hinting (e.g., PEP 526, PEP 544, PEP 585, PEP 604, PEP 695).

## Project-Specific Conventions and Preferences

In addition to the general Python guidelines, we follow these project-specific conventions:

1.  **Reduce Verbosity and Deep `if/else` Structures:**
    - Strive for concise and readable code.
    - Refactor deeply nested `if/else` blocks. Consider using techniques like guard clauses, dictionaries for dispatch, or polymorphism.

2.  **One Class Per File:**
    - Each class definition should reside in its own Python file.
    - The filename should typically match the class name (e.g., `MyClass` in `my_class.py`).

3.  **Exception Handling - Propagation:**
    - Allow general exceptions (`Exception`) to propagate up the call stack.
    - Avoid catching and re-raising general exceptions in lower-level functions solely for logging, unless it adds significant contextual information not available higher up.
    - Retain specific, meaningful exception handling (e.g., for `FloatingPointError`, `ValueError`, `TypeError`) where it adds value to program flow or error recovery.

4.  **Automated Code Formatting (Black):**
    - Use `black` for automated code formatting to ensure consistency.
    - Configure `black` to run via `uv run black .`.
    - Integrate `black` into a pre-commit hook to format code before each commit.

5.  **Meaningful Comments Only:**
    - Comments should explain *why* something is done, or clarify complex logic, rather than *what* is being done if the code is self-explanatory.
    - Avoid non-documenting comments, such as those indicating trivial changes (e.g., "removed this," "added that") or stating the obvious.
    - Docstrings (as per PEP 257) are crucial for documenting the purpose, arguments, and behavior of public APIs.
```
