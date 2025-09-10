# my_krml_25506751

A small ML utilities package for data processing and evaluation.

## Installation

### Option 1: Install from TestPyPI

```bash
$ pip install -i https://test.pypi.org/simple/ my_krml_25506751==2025.0.8.0
```
### Option 2: Install locally (after unzipping submission)

```bash
cd my_krml_25506751
pip install .
```

## Usage
```python

import numpy as np
import pandas as pd
from my_krml_25506751.models.performance import print_regressor_scores
from my_krml_25506751.data.sets import pop_target

# Create a simple dataframe
df = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5],
    "feature2": [5, 4, 3, 2, 1],
    "target":   [1.1, 1.9, 3.2, 4.1, 4.8]
})

# Split target from features
X, y = pop_target(df, target_col="target")

# Fake predictions for demo
y_preds = np.array([1, 2, 3, 4, 5])

# Print scores
print_regressor_scores(y_preds, y, set_name="Demo")
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`my_krml_25506751` was created by Parth T. Parth T retains all rights to the source and it may not be reproduced, distributed, or used to create derivative works.

## Credits

`my_krml_25506751` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
