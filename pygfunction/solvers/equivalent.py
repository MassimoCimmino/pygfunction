# -*- coding: utf-8 -*-
from time import perf_counter

import numpy as np
from scipy.cluster.hierarchy import (
    cut_tree,
    dendrogram,
    linkage
    )
from scipy.interpolate import interp1d as interp1d

from ._base_solver import _BaseSolver
from ..boreholes import _EquivalentBorehole
from ..heat_transfer import (
    finite_line_source,
    finite_line_source_equivalent_boreholes_vectorized,
    finite_line_source_vectorized
    )
from ..networks import _EquivalentNetwork


class Equivalent(_BaseSolver):
    """
    Equivalent solver for the evaluation of the g-function.

    This solver uses hierarchical agglomerative clustering to identify groups
    of boreholes that are expected to have similar borehole wall temperatures
    and heat extraction rates, as proposed by Prieto and Cimmino (2021)
    [#Equivalent-PriCim2021]_. Each group of boreholes is represented by a
    single equivalent borehole. The FLS solution is adapted to evaluate
    thermal interactions between groups of boreholes. This greatly reduces
    the number of evaluations of the FLS solution and the size of the system of
    equations to evaluate the g-function.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                **Uniform heat transfer rate**. This corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#Equivalent-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Equivalent-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#Equivalent-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Equivalent-Cimmin2019]_.

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    m_flow_borehole : (nInlets,) array or (nMassFlow, nInlets,) array, optional
        Fluid mass flow rate into each circuit of the network. If a
        (nMassFlow, nInlets,) array is supplied, the
        (nMassFlow, nMassFlow,) variable mass flow rate g-functions
        will be evaluated using the method of Cimmino (2024)
        [#Equivalent-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#Equivalent-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Equivalent-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral. When using the 'equivalent'
        solver, the approximation is only applied to the thermal response at
        the borehole radius. Thermal interaction between boreholes is evaluated
        using the FLS solution.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    kClusters : int, optional
        Increment on the minimum number of equivalent boreholes determined by
        cutting the dendrogram of the bore field given by the hierarchical
        agglomerative clustering method. Increasing the value of this parameter
        increases the accuracy of the method.
        Default is 1.

    References
    ----------
    .. [#Equivalent-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Equivalent-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
    .. [#Equivalent-Cimmin2019] Cimmino, M. (2019). Semi-analytical method
       for g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.
    .. [#Equivalent-PriCim2021] Prieto, C., & Cimmino, M.
       (2021). Thermal interactions in large irregular fields of geothermal
       boreholes: the method of equivalent borehole. Journal of Building
       Performance Simulation, 14 (4), 446-460.
    .. [#Equivalent-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#Equivalent-Cimmin2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def initialize(self, disTol=0.01, tol=1.0e-6, kClusters=1, **kwargs):
        """
        Initialize paramteters. Identify groups for equivalent boreholes.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        self.disTol = disTol
        self.tol = tol
        self.kClusters = kClusters
        # Check the validity of inputs
        self._check_solver_specific_inputs()
        # Initialize groups for equivalent boreholes
        nSources = self.find_groups()
        self.nBoreSegments = [self.nBoreSegments[0]] * self.nEqBoreholes
        self.segment_ratios = [self.segment_ratios[0]] * self.nEqBoreholes
        self.boreSegments = self.borehole_segments()
        self._i0Segments = [sum(self.nBoreSegments[0:i])
                            for i in range(self.nEqBoreholes)]
        self._i1Segments = [sum(self.nBoreSegments[0:(i + 1)])
                            for i in range(self.nEqBoreholes)]
        return nSources

    def thermal_response_factors(self, time, alpha, kind='linear'):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments in the borefield at all time steps using the finite line
        source solution.

        This method returns a scipy.interpolate.interp1d object of the matrix
        of thermal response factors, containing a copy of the matrix accessible
        by h_ij.y[:nSources,:nSources,:nt+1]. The first index along the
        third axis corresponds to time t=0. The interp1d object can be used to
        obtain thermal response factors at any intermediate time by
        h_ij(t)[:nSources,:nSources].

        Parameters
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        kind : string, optional
            Interpolation method used for segment-to-segment thermal response
            factors. See documentation for scipy.interpolate.interp1d.
            Default is linear.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...',
                  end='')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Initialize chrono
        tic = perf_counter()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        segment_lengths = self.segment_lengths()

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        # Groups correspond to unique pairs of borehole dimensions
        for pairs in self.borehole_to_borehole:
            i, j = pairs[0]
            # Prepare inputs to the FLS function
            dis, wDis = self._find_unique_distances(self.dis, pairs)
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, j)
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            N2 = np.array(
                [[self.boreholes[j].nBoreholes for (i, j) in pairs]]).T
            # Evaluate FLS at all time steps
            h = finite_line_source_equivalent_boreholes_vectorized(
                time, alpha, dis, wDis, H1, D1, H2, D2, N2)
            # Broadcast values to h_ij matrix
            for k, (i, j) in enumerate(pairs):
                i_segment = self._i0Segments[i] + i_pair
                j_segment = self._i0Segments[j] + j_pair
                h_ij[j_segment, i_segment, 1:] = h[k, k_pair, :]
                if not i == j:
                    h_ij[i_segment, j_segment, 1:] = (h[k, k_pair, :].T \
                        * segment_lengths[j_segment]/segment_lengths[i_segment]).T

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions
        # ---------------------------------------------------------------------
        # Groups correspond to unique borehole dimensions
        for group in self.borehole_to_self:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs(i, i)
            # Evaluate FLS at all time steps
            dis = self.boreholes[i].r_b
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            h = finite_line_source_vectorized(
                time, alpha, dis, H1, D1, H2, D2,
                approximation=self.approximate_FLS, N=self.nFLS)
            # Broadcast values to h_ij matrix
            for i in group:
                i_segment = self._i0Segments[i] + i_pair
                j_segment = self._i0Segments[i] + j_pair
                h_ij[j_segment, i_segment, 1:] = \
                    h_ij[j_segment, i_segment, 1:] + h[0, k_pair, :]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)), h_ij,
                        kind=kind, copy=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def find_groups(self, tol=1e-6):
        """
        Identify groups of boreholes that can be represented by a single
        equivalent borehole for the calculation of the g-function.

        Hierarchical agglomerative clustering is applied to the superposed
        steady-state finite line source solution (i.e. the steady-state
        dimensionless borehole wall temperature due to a uniform heat
        extraction equal for all boreholes). The number of clusters is
        evaluated by cutting the dendrogram at the half-height of the longest
        branch and incrementing the number of intercepted branches by the value
        of the kClusters parameter.

        Parameters
        ----------
        tol : float
            Tolerance on the temperature to identify the maxiumum number of
            equivalent boreholes.
            Default is 1e-6.

        Returns
        -------
        nSources : int
            Number of heat sources in the bore field.

        """
        if self.disp: print('Identifying equivalent boreholes ...', end='')
        # Initialize chrono
        tic = perf_counter()

        # Temperature change of individual boreholes
        self.nBoreholes = len(self.boreholes)
        # Equivalent field formed by all boreholes
        eqField = _EquivalentBorehole(self.boreholes)
        if self.nBoreholes > 1:
            # Spatial superposition of the steady-state FLS solution
            data = np.sum(finite_line_source(np.inf, 1., self.boreholes, self.boreholes), axis=1).reshape(-1,1)
            # Split boreholes into groups of same dimensions
            unique_boreholes = self._find_unique_boreholes(self.boreholes)
            # Initialize empty list of clusters
            self.clusters = []
            self.nEqBoreholes = 0
            for group in unique_boreholes:
                if len(group) > 1:
                    # Maximum temperature
                    maxTemp = np.max(data[group])
                    # Hierarchical agglomerative clustering based on temperatures
                    clusterization = linkage(data[group], method='complete')
                    dcoord = np.array(
                        dendrogram(clusterization, no_plot=True)['dcoord'])
                    # Maximum number of clusters
                    # Height to cut each tree to obtain the minimum number of clusters
                    disLeft = dcoord[:,1] - dcoord[:,0]
                    disRight = dcoord[:,2] - dcoord[:,3]
                    if np.max(disLeft) >= np.max(disRight):
                        i = disLeft.argmax()
                        height = 0.5*(dcoord[i,1] + dcoord[i,0])
                    else:
                        i = disRight.argmax()
                        height = 0.5*(dcoord[i,2] + dcoord[i,3])
                    # Find the number of clusters and increment by kClusters
                    # Maximum number of clusters
                    nClustersMax = min(np.sum(dcoord[:,1] > tol*maxTemp) + 1,
                                       len(group))
                    # Optimal number of cluster
                    nClusters = np.max(
                        cut_tree(clusterization, height=height)) + 1
                    nClusters = min(nClusters + self.kClusters, nClustersMax)
                    # Cut the tree to find the borehole groups
                    clusters = cut_tree(
                        clusterization, n_clusters=nClusters)
                    self.clusters = self.clusters + \
                        [label + self.nEqBoreholes for label in clusters]
                else:
                    nClusters = 1
                    self.clusters.append(self.nEqBoreholes)
                self.nEqBoreholes += nClusters
        else:
            self.nEqBoreholes = self.nBoreholes
            self.clusters = range(self.nBoreholes)
        # Overwrite boreholes with equivalent boreholes
        self.boreholes = [_EquivalentBorehole(
            [borehole
             for borehole, cluster in zip(self.boreholes, self.clusters)
             if cluster==i])
            for i in range(self.nEqBoreholes)]
        self.wBoreholes = np.array([b.nBoreholes for b in self.boreholes])
        # Find similar pairs of boreholes
        self.borehole_to_self, self.borehole_to_borehole = \
            self._find_axial_borehole_pairs(self.boreholes)
        # Store unique distances in the bore field
        self.dis = eqField.unique_distance(eqField, self.disTol)[0][1:]

        if self.boundary_condition == 'MIFT':
            pipes = [self.network.p[self.clusters.index(i)]
                     for i in range(self.nEqBoreholes)]
            self.network = _EquivalentNetwork(
                self.boreholes,
                pipes,
                nSegments=self.nBoreSegments[0],
                segment_ratios=self.segment_ratios[0])

        # Stop chrono
        toc = perf_counter()
        if self.disp:
            print(f' {toc - tic:.3f} sec')
            print(f'Calculations will be done using {self.nEqBoreholes} '
                  f'equivalent boreholes')

        return self.nBoreSegments[0]*self.nEqBoreholes

    def segment_lengths(self):
        """
        Return the length of all segments in the bore field.

        The segments lengths are used for the energy balance in the calculation
        of the g-function. For equivalent boreholes, the length of segments
        is multiplied by the number of boreholes in the group.

        Returns
        -------
        H : array
            Array of segment lengths (in m).

        """
        # Borehole lengths
        H = np.array([seg.H*seg.nBoreholes
                      for (borehole, nSegments, ratios) in zip(
                              self.boreholes,
                              self.nBoreSegments,
                              self.segment_ratios)
                      for seg in borehole.segments(
                              nSegments, segment_ratios=ratios)],
                     dtype=self.dtype)
        return H

    def _compare_boreholes(self, borehole1, borehole2):
        """
        Compare two boreholes and checks if they have the same dimensions :
        H, D, and r_b.

        Parameters
        ----------
        borehole1 : Borehole object
            First borehole.
        borehole2 : Borehole object
            Second borehole.

        Returns
        -------
        similarity : bool
            True if the two boreholes have the same dimensions.

        """
        # Compare lengths (H), buried depth (D) and radius (r_b)
        if (abs((borehole1.H - borehole2.H)/borehole1.H) < self.tol and
            abs((borehole1.r_b - borehole2.r_b)/borehole1.r_b) < self.tol and
            abs((borehole1.D - borehole2.D)/(borehole1.D + 1e-30)) < self.tol):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for real sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        deltaD1 = pair1[1].D - pair1[0].D
        deltaD2 = pair2[1].D - pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of lengths in each pair
        equal_H = abs((pair1[0].H - pair1[1].H)/pair1[0].H) < self.tol
        # Equality of buried depths differences
        cond_deltaD = abs(deltaD1 - deltaD2)/abs(deltaD1 + 1e-30) < self.tol
        # Equality of buried depths differences if all boreholes have the same
        # length
        cond_deltaD_equal_H = abs((abs(deltaD1) - abs(deltaD2))/(abs(deltaD1) + 1e-30)) < self.tol
        if cond_H and (cond_deltaD or (equal_H and cond_deltaD_equal_H)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_image_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        sumD1 = pair1[1].D + pair1[0].D
        sumD2 = pair2[1].D + pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths sums
        cond_sumD = abs((sumD1 - sumD2)/(sumD1 + 1e-30)) < self.tol
        if cond_H and cond_sumD:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_realandimage_pairs(self, pair1, pair2):
        """
        Compare two pairs of boreholes or segments and return True if the two
        pairs have the same FLS solution for both real and mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        if (self._compare_real_pairs(pair1, pair2)
            and self._compare_image_pairs(pair1, pair2)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _find_axial_borehole_pairs(self, boreholes):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        borehole pairs to simplify the evaluation of the FLS solution.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.

        Returns
        -------
        borehole_to_self : list
            Lists of borehole indexes for each unique set of borehole
            dimensions (H, D, r_b) in the bore field.
        borehole_to_borehole : list
            Lists of tuples of borehole indexes for each unique pair of
            boreholes that share the same (pairwise) dimensions (H, D).

        """
        # Compare for the full (real + image) FLS solution
        compare_pairs = self._compare_realandimage_pairs

        nBoreholes = len(boreholes)
        borehole_to_self = []
        # Only check for similarities if there is more than one borehole
        if nBoreholes > 1:
            borehole_to_borehole = []
            for i, borehole_i in enumerate(boreholes):
                # Compare the borehole to all known unique sets of dimensions
                for k, borehole_set in enumerate(borehole_to_self):
                    m = borehole_set[0]
                    # Add the borehole to the group if a similar borehole is
                    # found
                    if self._compare_boreholes(borehole_i, boreholes[m]):
                        borehole_set.append(i)
                        break
                else:
                    # If no similar boreholes are known, append the groups
                    borehole_to_self.append([i])
                # Note : The range is different from similarities since
                # an equivalent borehole to itself includes borehole-to-
                # borehole thermal interactions
                for j, borehole_j in enumerate(boreholes[i:], start=i):
                    pair0 = (borehole_i, borehole_j) # pair
                    pair1 = (borehole_j, borehole_i) # reciprocal pair
                    # Compare pairs of boreholes to known unique pairs
                    for pairs in borehole_to_borehole:
                        m, n = pairs[0]
                        pair_ref = (boreholes[m], boreholes[n])
                        # Add the pair (or the reciprocal pair) to a group
                        # if a similar one is found
                        if compare_pairs(pair0, pair_ref):
                            pairs.append((i, j))
                            break
                        elif compare_pairs(pair1, pair_ref):
                            pairs.append((j, i))
                            break
                    # If no similar pairs are known, append the groups
                    else:
                        borehole_to_borehole.append([(i, j)])
        else:
            # Outputs for a single borehole
            borehole_to_self = [[0]]
            borehole_to_borehole = [[(0, 0)]]
        return borehole_to_self, borehole_to_borehole

    def _find_unique_boreholes(self, boreholes):
        """
        Find unique sets of dimensions (h, D, r_b) in the bore field.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.

        Returns
        -------
        unique_boreholes : list
            List of list of borehole indices that correspond to unique
            borehole dimensions (H, D, r_b).

        """
        unique_boreholes = []
        for i, borehole_1 in enumerate(boreholes):
            for group in unique_boreholes:
                borehole_2 = boreholes[group[0]]
                # Add the borehole to a group if similar dimensions are found
                if self._compare_boreholes(borehole_1, borehole_2):
                    group.append(i)
                    break
            else:
                # If no similar boreholes are known, append the groups
                unique_boreholes.append([i])

        return unique_boreholes

    def _find_unique_distances(self, dis, indices):
        """
        Find the number of occurrences of each unique distances between pairs
        of boreholes.

        Parameters
        ----------
        dis : array
            Array of unique distances (in meters) in the bore field.
        indices : list
            List of tuples of borehole indices.

        Returns
        -------
        dis : array
            Array of unique distances (in meters) in the bore field.
        wDis : array
            Array of number of occurences of each unique distance for each
            pair of equivalent boreholes in indices.

        """
        wDis = np.zeros((len(dis), len(indices)), dtype=np.uint)
        for k, pair in enumerate(indices):
            i, j = pair
            b1, b2 = self.boreholes[i], self.boreholes[j]
            # Generate a flattened array of distances between boreholes i and j
            if not i == j:
                dis_ij = b1.distance(b2).flatten()
            else:
                # Remove the borehole radius from the distances
                dis_ij = b1.distance(b2)[
                    ~np.eye(b1.nBoreholes, dtype=bool)].flatten()
            wDis_ij = np.zeros(len(dis), dtype=np.uint)
            # Get insert positions for the distances
            iDis = np.searchsorted(dis, dis_ij, side='left')
            # Find indexes where previous index is closer
            prev_iDis_is_less = ((iDis == len(dis))|(np.fabs(dis_ij - dis[np.maximum(iDis-1, 0)]) < np.fabs(dis_ij - dis[np.minimum(iDis, len(dis)-1)])))
            iDis[prev_iDis_is_less] -= 1
            np.add.at(wDis_ij, iDis, 1)
            wDis[:,k] = wDis_ij

        return dis.reshape((1, -1)), wDis

    def _map_axial_segment_pairs(self, iBor, jBor,
                                 reaSource=True, imgSource=True):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        segment pairs along two boreholes to simplify the evaluation of the
        FLS solution.

        The returned H1, D1, H2, and D2 can be used to evaluate the segment-to-
        segment response factors using scipy.integrate.quad_vec.

        Parameters
        ----------
        iBor : int
            Index of the first borehole.
        jBor : int
            Index of the second borehole.

        Returns
        -------
        H1 : float
            Length of the emitting segments.
        D1 : array
            Array of buried depths of the emitting segments.
        H2 : float
            Length of the receiving segments.
        D2 : array
            Array of buried depths of the receiving segments.
        i_pair : list
            Indices of the emitting segments along a borehole.
        j_pair : list
            Indices of the receiving segments along a borehole.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).

        """
        # Initialize local variables
        borehole1 = self.boreholes[iBor]
        borehole2 = self.boreholes[jBor]
        assert reaSource or imgSource, \
            "At least one of reaSource and imgSource must be True."
        if reaSource and imgSource:
            # Find segment pairs for the full (real + image) FLS solution
            compare_pairs = self._compare_realandimage_pairs
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs
        # Dive both boreholes into segments
        segments1 = borehole1.segments(
            self.nBoreSegments[iBor], segment_ratios=self.segment_ratios[iBor])
        segments2 = borehole2.segments(
            self.nBoreSegments[jBor], segment_ratios=self.segment_ratios[jBor])
        # Prepare lists of segment lengths
        H1 = []
        H2 = []
        # Prepare lists of segment buried depths
        D1 = []
        D2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[iBor], dtype=np.uint),
                           self.nBoreSegments[jBor])
        j_pair = np.tile(np.arange(self.nBoreSegments[jBor], dtype=np.uint),
                         self.nBoreSegments[iBor])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[iBor] * self.nBoreSegments[jBor],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for i, segment_i in enumerate(segments1):
            for j, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
                    pair_ref = (segments1[m], segments2[n])
                    # Stop if a similar pair is found and assign the index
                    if compare_pairs(pair, pair_ref):
                        k_pair[p] = k
                        break
                # If no similar pair is found : add a new pair, increment the
                # number of unique pairs, and extract the associated buried
                # depths
                else:
                    k_pair[p] = nPairs
                    H1.append(segment_i.H)
                    H2.append(segment_j.H)
                    D1.append(segment_i.D)
                    D2.append(segment_j.D)
                    unique_pairs.append((i, j))
                    nPairs += 1
                p += 1
        return np.array(H1), np.array(D1), np.array(H2), np.array(D2), i_pair, j_pair, k_pair

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver object
        are what is expected.

        """
        assert type(self.disTol) is float and self.disTol > 0., \
            "The distance tolerance 'disTol' should be a positive float."
        assert type(self.tol) is float and self.tol > 0., \
            "The relative tolerance 'tol' should be a positive float."
        assert type(self.kClusters) is int and self.kClusters >= 0, \
            "The precision increment 'kClusters' should be a positive int."
        assert np.all(np.array(self.nBoreSegments, dtype=np.uint) == self.nBoreSegments[0]), \
            "Solver 'equivalent' can only handle equal numbers of segments."
        assert np.all([np.allclose(segment_ratios, self.segment_ratios[0]) for segment_ratios in self.segment_ratios]), \
            "Solver 'equivalent' can only handle identical segment_ratios for all boreholes."
        assert not np.any([b.is_tilted() for b in self.boreholes]), \
            "Solver 'equivalent' can only handle vertical boreholes."
        if self.boundary_condition == 'MIFT':
            assert np.all(np.array(self.network.c, dtype=int) == -1), \
                "Solver 'equivalent' is only valid for parallel-connected " \
                "boreholes."
            assert (self.m_flow_borehole is None
                    or (self.m_flow_borehole.ndim==1 and np.allclose(self.m_flow_borehole, self.m_flow_borehole[0]))
                    or (self.m_flow_borehole.ndim==2 and np.all([np.allclose(self.m_flow_borehole[:, i], self.m_flow_borehole[0, i]) for i in range(self.nBoreholes)]))), \
                "Mass flow rates into the network must be equal for all " \
                "boreholes."
            # Use the total network mass flow rate.
            if (type(self.network.m_flow_network) is np.ndarray and \
                len(self.network.m_flow_network)==len(self.network.b)):
                self.network.m_flow_network = \
                    self.network.m_flow_network[0]*len(self.network.b)
            # Verify that all boreholes have the same piping configuration
            # This is best done by comparing the matrix of thermal resistances.
            assert np.all(
                [np.allclose(self.network.p[0]._Rd, pipe._Rd)
                 for pipe in self.network.p]), \
                "All boreholes must have the same piping configuration."
        return
