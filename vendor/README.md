# vendor

`pyopengl-3.1.0-py3-none-any.whl` is PyOpenGL 3.1.0, which MuJoCo depends on.
The file published on PyPI is an sdist. Building it produces a wheel that also
contains `PyOpenGL-3.1.0.egg-info`. Current uv treats that file as a distutils
install and then cannot uninstall it, so `uv run` fails after `uv sync`.

This wheel is that build with the egg-info entry removed. The OpenGL package
contents are unchanged. It is BSD licensed (Mike C. Fletcher).
