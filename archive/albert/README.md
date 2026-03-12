# HST_UV

The images from projects 17179 and 16741 have been studied. Photometry has been performed in the UV spectrum using these Hubble images.

Check out this: https://archive.stsci.edu/proposal_search.php?id=17179&mission=hst

To accomplish this, the code created was "Code.ipynb".
In this code, multiple essential libraries for the photometric analysis of supernovae and their host galaxies had to be imported. The main libraries used include:

hostphot: Specialized in obtaining astronomical image cutouts and performing photometry on extended and point sources.

numpy: For efficient handling of numerical arrays.

matplotlib and aplpy: For visualizing FITS images and photometric data.

astropy.io.fits: For manipulating FITS files, the standard format in astronomy.

astropy.wcs: For handling celestial coordinate systems.

pandas: For managing tabular data related to photometric results.

pickle: For serializing and deserializing Python objects, allowing efficient data storage and retrieval.

shutil: For managing files and directories.

In the first part of the code, the function hms_dms_to_degrees is defined. Its purpose is to convert equatorial coordinates expressed in the Hours-Minutes-Seconds (HMS) and Degrees-Minutes-Seconds (DMS) systems into decimal degrees. This is crucial to ensure compatibility with astrometric analysis tools and astronomical databases, which require coordinates in this format.

The next part of the code utilizes hostphot.cutouts to download astronomical images from observational surveys. Key functionalities include:

Image downloading: Accessing astronomical databases to obtain image cutouts around the supernova’s position.
Configuration of Hubble Space Telescope (HST) images: Specifically, handling HST instrumentation to analyze data in the ultraviolet F275W filter.
Image stacking (coadd): Combining multiple exposures to improve the signal-to-noise ratio and detect fainter structures within the host galaxy.

Later, the hostphot.local_photometry module is used to obtain photometric magnitudes in different apertures. In local photometry, the flux is measured within a defined aperture around the supernova, allowing the estimation of the supernova's luminous contribution in different photometric bands.

The function plot_supernova_custom_sigma is implemented to visualize the supernova and its surroundings at different contrast scales, highlighting structures at various sigma (background standard deviation) levels. Its purpose is to facilitate the identification of the supernova within the host galaxy and assess potential photometric contamination. Its features include:

Contrast scale adjustment: By manipulating sigma levels, different structures in the image can be highlighted.

Aperture overlay: Circles or ellipses are drawn around the supernova to indicate the aperture used for photometry.

Marking of nearby sources: Identifying stars or bright regions that may affect photometry.

Compatibility with celestial coordinates: By integrating astropy.wcs, the image is projected with correct astronomical coordinates.

The code stores the photometric results in pickle files, enabling quick and efficient data retrieval without rerunning the analysis. Additionally, pandas is used for managing tabular data, facilitating visualization and further analysis.

The matplotlib and aplpy libraries are used to generate plots of the astronomical images.

Thus, throughout this code, hostphot has been used to automate photometry, enabling reproducible analysis of astronomical images. Celestial coordinate conversion has been performed, essential for compatibility with databases and astrometric tools. Then, image stacking techniques were applied, improving observation sensitivity and detecting low-surface-brightness structures. Finally, photometry calculations were performed in different apertures, and various plots of the studied images were generated.
