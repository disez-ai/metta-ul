# metta-ul: Clustering Algorithms in MeTTa

## Overview

metta-ul is a basic implementation of clustering algorithms in the MeTTa language. It includes implementations of:

- K-Means
- Gaussian Mixture Models (GMM)
- Spectral Clustering
- Hierarchical Clustering

This project is packaged as a Python module and includes a Dockerized environment for running tests using `pytest`.

## Authors

- **Ramin Barati** - <rekino@gmail.com>
- **Amirhossein Nourani Zadeh** - <amirhossein.nouranizadeh@gmail.com>
- **Farhoud** - <farhoud.m7@gmail.com>

## Requirements

- Python 3.7 or later
- Docker
- `hyperon` >= 0.2.2
- `scikit-learn`

## Installation

### Using pip

```sh
pip install -e .
```

### Using Docker

Build and run the containerized environment:

```sh
docker build . -t metta_ul
```

## Running Tests

### Running tests inside Docker

You can run tests using the provided `Makefile`. This will:

1. Build the Docker image
2. Run tests inside a container
3. Clean up the container after the test run

To execute:

```sh
make test
```

Alternatively, if you want to run pytest directly inside Docker:

```sh
docker run -it --rm --mount type=bind,src=$(pwd),dst=/app --name metta_ul_run metta_ul pytest -s
```

## Contributing

1. Fork the repository
2. Create a new branch (`feature-branch`)
3. Commit changes and push to your branch
4. Submit a pull request

## License

This project is licensed under the MIT License.
