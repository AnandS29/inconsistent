# inconsistent
Analysis of RLHF under inconsistent feedback

## Project structure

All Python code is under the `inconsistent_preferences` package. Run

    pip install -r requirements.txt

to install dependencies.

## Running reward learning

To train a reward function in the 1D environment, run

    python -m inconsistent_preferences.train_pref --env linear1d --stats --verbose

## Linting/formatting/type checking/testing

When commits are pushed to GitHub, they are checked with the [Flake8 linter](https://flake8.pycqa.org/en/latest/), [Black code formatter](https://black.readthedocs.io/en/stable/), [isort import sorter](https://pycqa.github.io/isort/index.html). and [Mypy type checker](http://mypy-lang.org/). Tests are also run with [pytest](https://docs.pytest.org/en/7.2.x/). To install these tools locally, run

    pip install --upgrade -r requirements_dev.txt

They can then be run using `./lint.sh`.
