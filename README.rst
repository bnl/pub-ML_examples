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

* ``tell`` : tell the agent about some data
* ``report`` : construct a report (message, visualization, etc.) about the data
* ``ask`` : ask the agent what to do next (for more see  `bluesky-adaptive <https://blueskyproject.io/bluesky-adaptive/>`_)


Unsupervised learning (Non-negative matrix factorization)
*********************************************************
The `NMF companion agent <bnl_ml/unsupervised/agent.py>`_ keeps a constant cache of data to perform the reduction on.
We treat these data as *dependent* variables, with *independent* variables coming fom the experiment.
In the case study presented, the independent variables are temperature measurements, and the dependent variables are the 1-d spectra.
Each call to ``report`` updates the decomposition using the full dataset, and updates the plots in the visualization.


the NMF companion agent is wrapped in a filesystem watcher, ``DirectoryAgent``, which monitors a directory periodically.
If there is new data in the target directory, the ``DirectoryAgent`` tells the NMF companion about the new data,
and triggers a new ``report``.


Anomaly detection
*****************

Supervised learning (Failure Classification)
********************************************


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

