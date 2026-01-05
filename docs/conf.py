# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import ast
import os
import sys
from pathlib import Path

import gmsh
import plotly.io as pio
import pyvista
from plotly.io._sg_scraper import plotly_sg_scraper
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper
from sphinx_gallery.sorting import FileNameSortKey

gmsh.initialize()

gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("General.Verbosity", 0)

pio.renderers.default = "sphinx_gallery"

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document_build")
# pyvista.set_jupyter_backend(None)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ["PYVISTA_BUILDING_GALLERY"] = "true"


this_dir = Path(__file__).resolve().parent.parent

about_path = this_dir / "opstool" / "__about__.py"
with open(about_path) as f:
    for line in f:
        if line.startswith("__version__"):
            __version__ = ast.literal_eval(line.split("=", 1)[1].strip())
            break

# include pkg root folder to sys.path
os.environ["PYTHONPATH"] = ":".join((str(this_dir), os.environ.get("PYTHONPATH", "")))
sys.path.append(str(this_dir))

project = "opstool"
copyright = "2025, Yexiang Yan"
author = "Yexiang Yan"
version = release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.autosummary",
    "nbsphinx",
    # "myst_nb",
    "jupyter_sphinx",
    # 'jupyter_sphinx.execute'
    "sphinx_copybutton",
    "furo.sphinxext",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
]

exclude_patterns = [
    "_build/**",
    "**/.ipynb_checkpoints/**",
    "_sphinx_gallery_examples/**/*.ipynb",
    "_sphinx_gallery_examples/**/*.py",
    "_sphinx_gallery_examples/**/*.py.md5",
    "_sphinx_gallery_examples/**/*.zip",
    "_sphinx_gallery_examples/**/*.codeobj.json",
    "Thumbs.db",
    ".DS_Store",
]

# autodoc config
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"

# napoleon config
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_ivar = True

# nbsphinx config
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]

# templates_path = ["_templates"]


sd_custom_directives = {
    "dropdown": {
        "inherit": "dropdown",
        "options": {
            "icon": "pencil",
            "class-container": "sn-dropdown-default",
        },
    }
}


# pyvista plot directive
# matplotlib plot directive
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["png"]

# pyvista plot directive
pyvista_plot_include_source = True


sphinx_gallery_conf = {
    "examples_dirs": ["quick-start", "../examples", "src/vis"],
    "gallery_dirs": [
        "_sphinx_gallery_examples/quick-start",
        "_sphinx_gallery_examples/examples",
        "_sphinx_gallery_examples/vis-guide",
    ],
    "image_scrapers": (DynamicScraper(), plotly_sg_scraper, "matplotlib"),
    "download_all_examples": False,
    "remove_config_comments": True,
    "reset_modules_order": "both",
    "filename_pattern": "ex-.*\\.py",
    "ignore_pattern": "_.*\\.py",
    "backreferences_dir": None,
    "pypandoc": True,
    "capture_repr": ("_repr_html_",),
    "within_subsection_order": FileNameSortKey,
    "nested_sections": True,
    "reset_modules": ("opstool.utils._reset_configs_for_doc",),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "sphinx_rtd_theme"
# html theme
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["_css/shared.css"]
html_css_files += ["_css/furo.css"]
# html_css_files += ["_css/collapse_output.css"]
html_favicon = "_static/logo.png"
html_theme_options = {
    "light_logo": "logo-light.png",  # add light mode logo
    "dark_logo": "logo-dark.png",  # add dark mode logo
    "sidebar_hide_name": True,  # hide the name of a project in the sidebar (already in logo)
    "source_repository": "https://github.com/yexiang92/opstool",
    "source_branch": "master",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        # "font-stack": "Arial, sans-serif",
        "font-stack--monospace": "Monaco, monospace",
        # "font-stack--headings": "Georgia, serif",
    },
}
templates_path = ["_static/_templates/furo", "_static/_templates"]
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "side-github.html",
        "sidebar/variant-selector.html",
    ]
}
html_context = {"repository": "yexiang92/opstool"}


pygments_style = "tango"
pygments_dark_style = "lightbulb"
