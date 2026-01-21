# Contributing to Tensorlink

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.3+
- Git

### Getting Started

1. Fork and clone the repository:
```bash
git clone https://github.com/mattjhawken/tensorlink.git
cd tensorlink
```

2. Install development dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

Run the full test suite:
```bash
pytest
```

### Code Quality

Before committing, run:
```bash
pre-commit run -a
```

This runs:
- Black (code formatting)
- Flake8 (linting)
- isort (import sorting)
- Type checking (if applicable)

### Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run pre-commit: `pre-commit run -a`
5. Commit with clear messages
6. Push and create a PR

## Guidelines

- Write tests for new features
- Update documentation as needed
- Follow existing code style
- Keep PRs focused and atomic. Include:
  * A brief summary of your change
  * Related issue numbers (e.g., `Closes #42`)
  * Any relevant logs, test output, or screenshots

## Questions?

Join our [Discord](https://discord.gg/aCW2kTNzJ2) for development discussions!
