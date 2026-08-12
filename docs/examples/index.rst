:og:description: Worked examples for biotuner, from peak extraction to tunings, geometry and materials.

########
Examples
########

Every page here is a notebook you can run. They are grouped by what you are
trying to do rather than by which module does it.

If you are new, start with :doc:`peaks_extraction/peaks_extraction` — almost
everything else begins with a set of spectral peaks.


Start here
==========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Peaks extraction
      :link: peaks_extraction/peaks_extraction
      :link-type: doc

      Five ways to find the frequencies that matter in a signal, and how much
      they disagree.

   .. grid-item-card:: Plotting
      :link: plotting/plotting
      :link-type: doc

      The visualisation API, walked through on rich synthetic signals.

   .. grid-item-card:: Harmonicity metrics
      :link: harmonicity_metrics/harmonicity_metrics
      :link-type: doc

      How harmonic is a set of peaks? The measures, side by side.


Signals and spectra
===================

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: From an MNE epochs file
      :link: biotuner_MNE/biotuner_MNE
      :link-type: doc

      Harmonicity metrics computed straight off an existing MNE object.

   .. grid-item-card:: The harmonic spectrum
      :link: harmonic_spectrum/harmonic_spectrum
      :link-type: doc

      Harmonicity across the whole spectrum, not only at the peaks.

   .. grid-item-card:: Mathematical series
      :link: math_series/math_series
      :link-type: doc

      Which classic series — Fibonacci, primes, powers — does a signal follow?

   .. grid-item-card:: Resonance cookbook
      :link: resonance_cookbook/resonance_cookbook
      :link-type: doc

      A practical tour of resonance, harmonic spectrum and connectivity.

   .. grid-item-card:: Phase-amplitude coupling
      :link: phase_amplitude_coupling/phase_amplitude_coupling
      :link-type: doc

      Comodulograms: which phase drives which amplitude, and how strongly.


Music from signals
==================

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Tunings from biosignals
      :link: scale_construction/scale_construction
      :link-type: doc

      Turning a recording's peaks into a playable scale.

   .. grid-item-card:: Moment-of-symmetry scales — the short tour
      :link: mos/showcase
      :link-type: doc

      From the piano's white keys to fitting a scale to a biosignal, with
      audio. **Start here for MOS.**

   .. grid-item-card:: Moment-of-symmetry scales — the full tour
      :link: mos/mos
      :link-type: doc

      The labyrinth in depth: modes, temperaments, just intonation,
      trajectories, Fourier scratching.

   .. grid-item-card:: The MOS explorers
      :link: mos/explorers
      :link-type: doc

      Ten interactive widgets. Drag a generator and watch the scale follow.

   .. grid-item-card:: Spectral chords
      :link: spectral_chords/spectral_chords
      :link-type: doc

      Chords derived from the spectrum of a biosignal.

   .. grid-item-card:: Euclidean rhythms
      :link: rhythm_construction/rhythm_construction
      :link-type: doc

      Pulses distributed as evenly as a step count allows, driven by a signal.


Harmonic geometry
=================

Chords rendered as shape: closed-form curves, eigenmodes, and the metrics that
compare them.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Lissajous and harmonograph
      :link: harmonic_geometry/01_lissajous_and_harmonograph
      :link-type: doc

      Closed-form figures and damped harmonograph traces.

   .. grid-item-card:: Chladni and spherical harmonics
      :link: harmonic_geometry/02_chladni_and_spherical_harmonics
      :link-type: doc

      Eigenmodes of the wave equation on a bounded medium.

   .. grid-item-card:: Circles, polygons and cycloids
      :link: harmonic_geometry/03_circular_patterns
      :link-type: doc

      A ratio set wrapped onto a circle, once per equave.

   .. grid-item-card:: Fractals and L-systems
      :link: harmonic_geometry/04_fractal_and_generative
      :link-type: doc

      Deterministic fractal and generative layouts.

   .. grid-item-card:: Three dimensions
      :link: harmonic_geometry/05_three_dimensional
      :link-type: doc

      Harmonic knots and other 3-D generators.

   .. grid-item-card:: Metrics and transitions
      :link: harmonic_geometry/06_metrics_and_transitions
      :link-type: doc

      Comparing geometries, and morphing one chord into another.

   .. grid-item-card:: Chladni cymatics
      :link: harmonic_geometry/07_chladni_cymatics
      :link-type: doc

      Sand on a square plate, driven by a chord.


Media — chord-driven response operators
=======================================

Give a physical medium a chord and see what it does.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Protocol, domains and pipelines
      :link: harmonic_geometry_media/01_media_protocol_and_pipelines
      :link-type: doc

      How a medium is defined, and how media chain together.

   .. grid-item-card:: Eigenmode and wave field
      :link: harmonic_geometry_media/02_eigenmode_and_wave_field
      :link-type: doc

      Standing waves on plates, spheres, crystals and pressure fields.

   .. grid-item-card:: Parametric and transport
      :link: harmonic_geometry_media/03_parametric_and_transport
      :link-type: doc

      Faraday instability, granular density, streamlines, streaming.

   .. grid-item-card:: Morphogenetic
      :link: harmonic_geometry_media/04_morphogenetic
      :link-type: doc

      Snowflake growth and Gray–Scott patterns shaped by a chord.


Colour and matter
=================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Bio-colour
      :link: biocolors/biocolors
      :link-type: doc

      A signal's spectrum, or a tuning's ratios, turned into a palette.

   .. grid-item-card:: Bio-elements
      :link: bioelements/bioelements
      :link-type: doc

      From a biosignal to the periodic table, and on to materials.


The toolbox paper
=================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Design and implementation
      :link: toolbox_paper_notebook/toolbox_paper_notebook
      :link-type: doc

      The examples that accompany the paper.

   .. grid-item-card:: Result figures
      :link: toolbox_paper_result_notebook/toolbox_paper_result_notebook
      :link-type: doc

      The figures, reproduced end to end.


.. Hidden toctrees. The cards above are the page; these give the sidebar its
   structure and let Sphinx resolve :doc: references. Captions group the
   entries in the left-hand navigation instead of one flat list.

.. toctree::
   :hidden:
   :caption: Start here

   peaks_extraction/peaks_extraction
   plotting/plotting
   harmonicity_metrics/harmonicity_metrics

.. toctree::
   :hidden:
   :caption: Signals and spectra

   biotuner_MNE/biotuner_MNE
   harmonic_spectrum/harmonic_spectrum
   math_series/math_series
   resonance_cookbook/resonance_cookbook
   phase_amplitude_coupling/phase_amplitude_coupling

.. toctree::
   :hidden:
   :caption: Music from signals

   scale_construction/scale_construction
   mos/showcase
   mos/mos
   mos/explorers
   spectral_chords/spectral_chords
   rhythm_construction/rhythm_construction

.. toctree::
   :hidden:
   :caption: Harmonic geometry

   harmonic_geometry/01_lissajous_and_harmonograph
   harmonic_geometry/02_chladni_and_spherical_harmonics
   harmonic_geometry/03_circular_patterns
   harmonic_geometry/04_fractal_and_generative
   harmonic_geometry/05_three_dimensional
   harmonic_geometry/06_metrics_and_transitions
   harmonic_geometry/07_chladni_cymatics

.. toctree::
   :hidden:
   :caption: Media

   harmonic_geometry_media/01_media_protocol_and_pipelines
   harmonic_geometry_media/02_eigenmode_and_wave_field
   harmonic_geometry_media/03_parametric_and_transport
   harmonic_geometry_media/04_morphogenetic

.. toctree::
   :hidden:
   :caption: Colour and matter

   biocolors/biocolors
   bioelements/bioelements

.. toctree::
   :hidden:
   :caption: The toolbox paper

   toolbox_paper_notebook/toolbox_paper_notebook
   toolbox_paper_result_notebook/toolbox_paper_result_notebook
