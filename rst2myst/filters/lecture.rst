.. _cass_koopmans_1:

.. raw:: html

		<div id="qe-notebook-header" align="right" style="text-align:right;">

			<a href="https://quantecon.org/" title="quantecon.org">
				<img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
			</a>
			
		</div>


*************************************
Main header
*************************************

.. contents:: :depth: 2

Overview
=========

This lecture and in :doc:`Cass-Koopmans Competitive Equilibrium <cass_koopmans_2>` describe a model that Tjalling Koopmans :cite:`Koopmans`
and David Cass :cite:`Cass` used to analyze optimal growth.

.. _ex2:

The Model
==================

Time is discrete and takes values :math:`t = 0, 1 , \ldots, T` where :math:`T` is  finite.

(We'll study a limiting case in which  :math:`T = + \infty` before concluding).

Planning Problem
=================

A planner chooses an allocation :math:`\{\vec{C},\vec{K}\}` to
maximize :eq:`utility-functional` subject to :eq:`allocation`.

.. _first_order:

First-order necessary conditions
---------------------------------

We now compute **first order necessary conditions** for extremization of the Lagrangian:


Setting Initial Capital to Steady State Capital
================================================

When  :math:`T \rightarrow +\infty`, the optimal allocation converges to
steady state values of :math:`C_t` and :math:`K_t`.
