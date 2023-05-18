# System rekomendacyjny oparty o model filtrowania kolaboratywnego

## Table of contents
* [Logic applied in application](#Logic-applied-in-application)
* [Technologies](#Technologies)
* [Requirmenets](#Requirements)
* [Examples](#Examples)
* [External sources](#External sources)

## Logic applied in application
Model uczenia maszynowego - system rekomendacyjny oparty o model filtrowania kolaboratywnego.

## Technologies

* Python version: 3.10.10
* LightFM version: 1.17
* Jupyter notebook version: 6.5.4

## Requirements

Wszystkie wymagania zostały wylistowane w pliku `requirements.txt`.

## Setup and istallations

To run the application, install setupy.py:

```
sudo python3 setup.py --install
```

## Examples
Run `recommender_system.ipynb` to train the models. Amend parameters like number of epochs or alpha value for experiments.

Run `model_analysis.ipynb` to analyze models AUC and duration.

### AUC metrics comparision

![AUC](../AUC.png)

### Duration comparision

![Duration](../duration.png)

Run `recommendations_generation.ipynb` to retrieve predictions for all or given users.

## External sources

* [LightFM library documentation](https://making.lyst.com/lightfm/docs/home.html)
