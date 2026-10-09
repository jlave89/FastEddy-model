========================================================
Real-world downscaled FastEddy simulation with buildings
========================================================

This tutorial involves setting up a real-world downscaled simulation that includes resolved buildings and closely follows the same procedure outlined in section 4.1. Additional steps for including buildings are described below and all required input datasets to run this tutorial are provided in this `Zenodo record <https://zenodo.org/records/17419234>`_.

In the *GeoSpec* preprocessing step, and additional 2d field describing building heights above ground level is required in the georeference input NetCDF file.

.. code-block:: none

   float BuildingHeights(y, x) ;

This tutorial provides an example georeference input file with building heights for downtown Dallas, TX (:code:`Dallas_input_Oct2025_lod13.nc`). The *geospec.json* parameter file option :code:`urban_opt : 1` needs to be selected for building height information to be ingested in the reference standard-format NetCDF output file upon execution of **GeoSpec.py**.

The same :code:`urban_opt : 1` option needs must be included in the subsequent *SimGrid* and *GenICBCs* stages, where building information is used for the creation of a :code:`BuildingMask` array containing gridded information of building presence, and ensuring that winds, subgrid-scale TKE and hydrometeors are set to zero within buildings for both initial and boundary conditions. In this example *center_lat* and *center_lon* are (32.7835, -96.8092), required by **simgrid.json**.

After initial and boundary conditions have been created, a building-resolving FastEddy simulation can be undertaken by activating the urban model capability in the parameters file (see **tutorials/examples/Example10_REALCASE_Dallas_urban.in**).

.. code-block:: none

   #--URBAN
   urbanSelector = 1 # urban selector: 0=off, 1=on

The urban model capability has been implemented into FastEddy as an extension module, and is not compiled by default. To include the URBAN module in a build of FastEddy use the following compile flag:

.. code-block:: none

   make WITH_URBAN=1

The model used to represent buildings follows the immersed body force approach described in *Muñoz-Esparza et al., 2020*, and the tutorial case corresponds to the passage of a cold front (*Muñoz-Esparza et al. (2021, 2025)*. The figure below shows instantaneous wind speed and vertical velocity fields corresponding to 30min hindcast valid at 1500 UTC on November 11th 2019 (pre-frontal conditions). These horizontal contours are from the model's third vertical level, located at approximately 23 m above ground level.

.. image:: ../images/URBAN_tutorial_nz2_2panel.png
  :width: 900
  :alt: Alternative text

Full citation references can be found in the :doc:`Publications <../../publications>` section.

Surface Heat Flux Redistribution
--------------------------------

The Surface Heat Flux Redistribution (SHFR) option is included in the FastEddy URBAN module and therefore requires the urban capability enabled (:code:`urban_opt : 1`). When buildings are explicitly represented in FastEddy, surface sensible and latent heat fluxes are masked to zero over building-covered grid cells. This masking can reduce the total heat input in densely built areas.

The SHFR option compensates for this effect by redistributing the suppressed surface heat flux over the surrounding non-building grid cells. The redistribution is computed during the preprocessing stage with **SimGrid.py** and stored as a 2d field named :code:`UrbanHeatRedis`. During the FastEddy simulation, this factor is applied multiplicatively to the surface heat fluxes.

When buildings are explicitly resolved, using conventional urban roughness lengths derived for unresolved urban canopies may lead to a double counting of building-induced drag. To avoid this issue, SHFR is designed to operate together with a modified "street-like" roughness length for urban land-cover categories. This reduced roughness represents the aerodynamic properties of streets and open urban surfaces. The modified roughness values must therefore be used during the execution of **GeoSpec.py** (see section 3.1.1).

The SHFR algorithm accounts for the difference between fluxes computed with the original land-cover roughness and those computed with the modified street-like roughness. For this reason, SHFR requires a land-cover table with both values: *z0* and *z0urbanLES*. *z0urbanLES* must be set to 0.0 for non-urban categories.

To create the :code:`UrbanHeatRedis` field, add the following entries to the **simgrid.json** file:

.. code-block:: none

        "urban_heatRedis_opt": 1,
        "landcover_table": "/path_to_landcover_table/landcover_table.csv"

Note that `landcover_table` entry is the same as the landcover table specified in **geospec.json**. If :code:`urban_heatRedis_opt : 0`, no redistribution is applied and the :code:`landcover_table` entry can be left empty.

To run a building-resolving FastEddy simulation with SHFR, the capability must also be activated in the FastEddy parameters file:

.. code-block:: none

        #-- URBAN
        urban_heatRedis = 1 # selector to activate surface heat redistribution

Building temperature follows the surface boundary condition
-----------------------------------------------------------

With :code:`urbanSelector = 1` the building mask stays at its initial temperature, and with :code:`urbanSelector = 2` it is relaxed toward the fixed base state. Over a multi-hour run that solid becomes a thermal and pressure anomaly relative to the surroundings, and a spurious city-scale inflow or outflow develops. :code:`urbanSelector = 3` is the setting to use when that flow appears.

Selector 3 keeps the building drag and still suppresses thermal, density, moisture, and scalar tendencies inside the mask. After each completed timestep the building temperature is set from the same surface thermal boundary condition that forces the rest of the domain. For :code:`surflayerSelector = 2` or :code:`3` that target is the local skin temperature, converted to potential temperature with the base-state Exner function. For :code:`surflayerSelector = 1` the boundary condition is a heat flux: one potential temperature per footprint is integrated with :code:`urban_thermal_depth` (meters, default 1000) and applied through the column. Real-case runs use :code:`surflayerSelector = 3` and do not use that depth. Selector 3 requires the surface layer to be on.

Keep :code:`urban_heatRedis = 0` with selector 3. The building volume is the thermal reservoir, and redistribution would add that heat a second time through the streets. The checked-in Dallas namelist stays at :code:`urbanSelector = 1`, which is the published tutorial configuration.

.. code-block:: none

   #--URBAN
   urbanSelector = 3 # building temperature follows the surface thermal boundary condition
   urban_heatRedis = 0 # keep off with urbanSelector = 3
