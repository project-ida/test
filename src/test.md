---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="2a8027d5-be26-422a-b3ce-35fcdb9df4b3" -->
<a href="https://colab.research.google.com/github/project-ida/test/blob/test/test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/test/blob/test/test.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QBrOpoOm4p3r" outputId="bdb8ac2f-aba5-4ea4-f3bf-6e3e89dca62a"
print("hi")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 1000} id="MTnVX3ST41M5" outputId="ab1d481c-4b0e-4cda-c105-49b4cb4dfb8a"
import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x**2

# Line Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y1, label='sin(x)', linestyle='-', marker='o')
plt.plot(x, y2, label='cos(x)', linestyle='--', marker='x')
plt.title('Line Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
sizes = np.random.rand(50) * 100
colors = np.random.rand(50)

plt.figure(figsize=(8, 5))
plt.scatter(x_scatter, y_scatter, s=sizes, c=colors, alpha=0.7, cmap='viridis')
plt.title('Scatter Plot Example')
plt.xlabel('Random X Values')
plt.ylabel('Random Y Values')
plt.colorbar(label='Colour Intensity')
plt.grid(True)
plt.show()

# Bar Plot
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 45, 56, 78]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color='skyblue')
plt.title('Bar Plot Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.grid(axis='y')
plt.show()

# Histogram
data = np.random.normal(0, 1, 1000)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.title('Histogram Example')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

```python id="3lZygjk96lyA"

```
