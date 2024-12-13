---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="3f826b5b-1cf7-45b2-9622-89c10dbf1eb2" -->
<a href="https://colab.research.google.com/github/project-ida/test/blob/test/matt-test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/test/blob/test/matt-test.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="a0c58e6c-2dcf-4992-8d16-db9ec301f4b4" -->
## Test notebook
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9b2ae39f-1975-4c39-afb9-53b12f6e2088" outputId="b72570ae-04f6-4aca-c987-34ff83ef4093"
print("Hello World!")
```

```python id="n48g21aKNCg2"
print("branch test")
```

```python id="hh6eSMFS1dDw" outputId="62fd0039-71c6-4826-c1ff-bb9d4b268c66" colab={"base_uri": "https://localhost:8080/", "height": 472}
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plot
plt.plot(x, y, marker='o')
plt.title('Quick Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

```

```python id="9PwvhEhu1dqc" outputId="eba01fa2-35b1-4732-ed59-742338105fc5" colab={"base_uri": "https://localhost:8080/", "height": 472}
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plot
plt.plot(x, np.sin(x), marker='o')
plt.title('Quick Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

```

```python id="16e1ukZaIweV"

```
