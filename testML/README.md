# spectroscopy-ML
Machine learning for spectroscopy

**Disclaimer:** This is a work in progress


## Installation
`pip install git+https://github.com/MariaTsantaki/spectroscopy-ML`


## Usage

```python
import specML
from specML import Data, Model, Minimizer

data = specML.get_data()  # Data type
model = specML.get_model(data)  # Model type

# Get a random spectrum
flux = data.y.sample(1)
minimizer = Minimizer(flux, model)
res = minimizer.minimize(method='Nelder-Mead')

# See the results
print(res)

# Plot the results
minimizer.plot()
```

All `scipy.optimize.minimize` methods are available (although not all will work).
The minimization works by doing a chi squared minimization between the flux and
the calculated flux from, which is done with Machine Learning.
