***********************************************************************************************
Machine learning enabling high-throughput and remote operations at large-scale user facilities.
***********************************************************************************************
Overview
========



Abstract
********
Placeholder


Explanation of Examples
=======================
As with all things at a user facility, each model is trained or set-up according to the needs of the user and their science.
What is consistent across all AI agents, is their final communication paradigm.
The agent loads and stores the model and/or necessary data, and has at minimum the following methods.

* ``tell`` : tell the agent about some new data
* ``report`` : construct a report (message, visualization, etc.) about the data
* ``ask`` : ask the agent what to do next (for more see  `bluesky-adaptive <https://blueskyproject.io/bluesky-adaptive/>`_)


Unsupervised learning (Non-negative matrix factorization)
*********************************************************
The `NMF companion agent <bnl_ml/unsupervised/agent.py>`_ keeps a constant cache of data to perform the reduction on.
We treat these data as *dependent* variables, with *independent* variables coming fom the experiment.
In the case study presented, the independent variables are temperature measurements, and the dependent variables are the 1-d spectra.
Each call to ``report`` updates the decomposition using the full dataset, and updates the plots in the visualization.


The NMF companion agent is wrapped in a filesystem watcher, ``DirectoryAgent``, which monitors a directory periodically.
If there is new data in the target directory, the ``DirectoryAgent`` tells the NMF companion about the new data,
and triggers a new ``report``.

The construction of these objects, training, and visualization are all contained in the `run_unsupervised file <example_scripts/run_unsupervised.py>`_
and mirrored in the `corresponding notebook <example_scripts/run_unsupervised.ipynb>`_.

Anomaly detection
*****************
The model attributes a new observation to either normal or anomalous time series by comparing it to a large courpus of data collected at the beamline over an extended period of time. The development and updating of the model is done offline. Due to the nature of exparimental measurements, anomalous observatons may constitute a sizable portion of data withing a single collection period. Thus, a labeling of the data is required prior to model training. Once the model is trained it is saved as a binary file and loaded each time when ``AnomalyAgent`` is initialized.

A set of features devired from the original raw data, allowing the model to process time series of arbitary length.

The training can be found at `run_anomaly.py <example_scripts/run_anomaly.py>`_ with example deployment
infrastructure at `deploy_anomaly.py <example_scripts/deploy_anomaly.py>`_.

Supervised learning (Failure Classification)
********************************************
The classifications of failures involves training the models entirely offline.
This allows for robust model selection and specific deployment.
A suite of models from scikit-learn are trained and tested, with the most promising model chosen to deploy.
Since the models are lightweight, we re-train them at each instantiation during deployment with the most current dataset.
For deep learning models, it would be appropriate to save and version the weights of a model, can construct the model at
instantiation and load the weights.

The training can be found at `run_supervised.py <example_scripts/run_supervised.py>`_ with example deployment
infrastructure at `deploy_supervised.py <example_scripts/deploy_supervised.py>`_.
How this is implemented at the BMM beamline can be found concisely
`here <https://github.com/NSLS-II-BMM/profile_collection/blob/master/startup/BMM/xafs.py#L1167-L1169>`_,
where a wrapper agent does pointwise evaluation on UIDs of a document stream, using the ``ClassificationAgent``'s ``tell``--``report`` interface.


System Requirements
===================


Hardware Requirements
*********************


Software Requirements
*********************

OS Requirements
---------------
This package has been tested exclusively on Linux operating systems.

- Ubuntu 18.04
- PopOS 20.04

Python dependencies
-------------------

Getting Started
===============

Installation guide
******************


Install from github::

    $ python3 -m venv pub_env
    $ source pub_env/bin/activate

