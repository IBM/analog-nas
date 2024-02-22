Installation
============

The preferred way to install this package is by using the `Python package index`_::

    pip install analogainas

For ease of installation, install aihwkit library separately, refer to `AIHWKit installation`_:

The package require the following runtime libraries to be installed in your
system:

* `OpenBLAS`_: 0.3.3+
* `CUDA Toolkit`_: 9.0+ (only required for the GPU-enabled simulator [#f1]_)

.. [#f1] Note that GPU support is not available in OSX, as it depends on a
   platform that has official CUDA support.
   
.. _AIHWKit installation: https://aihwkit.readthedocs.io/en/latest/install.html
.. _Python package index: https://pypi.org/project/analogainas/
.. _OpenBLAS: https://www.openblas.net
.. _CUDA Toolkit: https://developer.nvidia.com/accelerated-computing-toolkit