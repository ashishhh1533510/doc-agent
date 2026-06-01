# TaskTracker

TaskTracker is a small in-memory task management library.

## Overview

This project provides a simple, in-memory system for managing tasks. It includes data models for tasks, a store for managing collections of tasks, and utilities for formatting and reporting on them.

The core modules are:

*   **`tasktracker.models`**: Defines the data models for tasks, including the `Task` object and its `Status` enumeration.
*   **`tasktracker.store`**: Provides an in-memory storage mechanism for tasks, along with methods for adding, querying, and updating tasks.
*   **`tasktracker.utils`**: Contains helper functions for formatting tasks into human-readable strings and summarizing task collections.

## Usage / API Reference

### `tasktracker.models`

Data models for tasks: the `Task` object and its `Status` enum.

#### `class Status(Enum)`

The lifecycle state of a task.

#### `class Task`

A single unit of work with a title, status, and optional due date.

```python
__init__(self, title: str, due: date | None = None, priority: int = 3)
```

Create a task with a title, an optional due date, and a priority (1 = highest).

```python
mark_done(self) -> None
```

Mark this task as completed.

```python
@property
is_overdue(self) -> bool
```

Return `True` if the task has a past due date and is not yet done.

### `tasktracker.store`

In-memory storage and querying for tasks.

#### `class TaskStore`

An in-memory collection of tasks with simple query helpers.

```python
__init__(self)
```

Create an empty task store.

```python
add(self, task: Task) -> None
```

Add a task to the store.

```python
pending(self) -> list[Task]
```

Return all tasks that are not yet done, sorted by priority.

```python
complete_all(self) -> int
```

Mark every task as done and return how many were updated.

```python
async sync(self, *sources, timeout: float = 5.0) -> bool
```

Sync tasks from the given remote sources; return `True` on success.

### `tasktracker.utils`

Helper functions for formatting and reporting on tasks.

```python
format_task(task: Task, show_priority: bool = True) -> str
```

Return a human-readable one-line summary of a task.

```python
summarize(tasks: list[Task]) -> dict[str, int]
```

Count tasks by status and return the counts as a dictionary.