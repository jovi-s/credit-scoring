# Credit Scoring

A machine learning based credit scorecard for lending risk analysis.

## Description

The purpose of this repository is to create a credit scoring model for a potential lending company. Historical data is provided.
The business use case of this application aims to solve the following objectives:

1. Achieve an overall default rate of total portfolio to be below 2.5%, and to provide recommendations on the optimal credit score cutoff rate.
2. Create a credit score for each individual, which is validated, and to provide guidance on the next steps.
3. Create deciles by credit score and provide risk and default levels by deciles (and cumulative).
4. Provide confidence for credit scores and default rates by bin.

## Getting Started

### Dependencies

- Any libraries needed before running this program is provided in `requirements.txt`

### Installing

- cd to the directory where requirements.txt is located.
- Activate your virtualenv.
- Run: pip install -r requirements.txt in your shell.
- Modify path in cell 1 of `modelling.ipynb` to root of credit-scoring directory.

### Executing program

- The program can be run through the jupyter notebooks.

## Authors

[Jovinder Singh](https://www.linkedin.com/in/jovindersingh/)

## Version History

- 0.3 (TODO)
  - Create python scripts for inference
- 0.2 (TODO)
  - Change baseline model to logistic regression
  - Complete [3.](#description) and [4.](#description)
- 0.1
  - Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- [finlytics-hub](https://github.com/finlytics-hub/credit_risk_model)
