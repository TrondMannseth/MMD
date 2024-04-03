#
# genToyExamplesResults
#
import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt
import sys
import os
from scipy import linalg
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

import decomp

# Initialize class
chol = decomp.Cholesky()


def main():
    #
    # ==============
    # Case selection
    # ==============
    #
    figure = 2
    #figure = 3
    #figure = 4
    #figure = 5
    #figure = 6
    #figure = 7
    #
    # ==============
    # Initialization
    # ==============
    #
    plotPath = '..' + os.sep + 'results' + os.sep + 'Figure' + str(figure) + os.sep
    #
    # Make paths if they do not exist
    #
    if not os.path.exists(plotPath):
        os.makedirs(plotPath)
    #
    # File prefix
    #
    filePrefix = 'Fig'
    file = plotPath + filePrefix
    #
    credLow = 0.02 #Approximately two standard deviations
    credHigh = 1 - credLow
    #
    L = 10
    x_size = 2 ** L
    y_size = 1
    #
    aspect = 1
    angle = 0
    #
    # Multilevel
    #
    #
    # ==============
    # Prior ensemble
    # ==============
    #
    pvariance = 1
    pnumber = 100
    pvar_type = 'cub'
    #
    pmean = np.zeros(x_size)
    pvar_range = 25
    #
    # ====
    # Data
    # ====
    #
    dnumber = pnumber
    drealizplotno = 0
    #
    if figure == 2:
        dmean = pmean
        dvar_range = pvar_range
        dvariance = pvariance
        dvar_type = pvar_type
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        labelTwoReas = str(figure) + 'b'
        plotTwoReas(file, labelTwoReas, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        xtmarksMah = [4000, 6000, 8000, 10000, 12000, 14000]
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'd'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
    elif figure == 3:
        dmean = pmean
        dvariance = pvariance
        dvar_type = pvar_type
        xtmarksMah = [2000, 6000, 10000, 14000]
        #
        #-----------------
        # Plots 3a, 3c, 3e
        #-----------------
        #
        dvar_range = 50
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'e'
        pltType = 'stds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
        #
        #-----------------
        # Plots 3b, 3d, 3f
        #-----------------
        #
        dvar_range = 37.5
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'b'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'd'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'f'
        pltType = 'stds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
    elif figure == 4:
        dmean = pmean + 1 * np.ones(x_size)
        dvar_range = pvar_range
        dvariance = pvariance
        dvar_type = pvar_type
        xtmarksMah = [4000, 6000, 8000, 10000, 12000, 14000]
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Realization means
        #
        labelMeansReas = str(figure) + 'b'
        plotMeansReas(file, labelMeansReas, pmean, dmean)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'd'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
    elif figure == 5:
        dvar_range = pvar_range
        dvariance = 0.01 * pvariance
        dvar_type = pvar_type
        #
        #-----------------
        # Plots 5a, 5c, 5e
        #-----------------
        #
        dmeanLeft = - 1.5
        dmeanRight = 1.5
        dmean = pmean + np.linspace(dmeanLeft, dmeanRight, x_size)
        xtmarksMah = [2000, 6000, 10000, 14000]
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'e'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
        #
        #-----------------
        # Plots 5b, 5d, 5f
        #-----------------
        #
        dmeanLeft = - 2
        dmeanRight = 2
        dmean = pmean + np.linspace(dmeanLeft, dmeanRight, x_size)
        xtmarksMah = [4000, 6000, 8000, 10000, 12000, 14000]
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'b'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'd'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'f'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
    elif figure == 6:
        dmeanLeft = - 2
        dmeanRight = 2
        dmean = pmean + np.linspace(dmeanLeft, dmeanRight, x_size)
        dvar_range = pvar_range
        dvar_type = pvar_type
        xtmarksMah = [4000, 6000, 8000, 10000, 12000, 14000]
        #
        #-----------------
        # Plots 6a, 6c, 6e
        #-----------------
        #
        dvariance = 0.1 * pvariance
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'e'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
        #
        #-----------------
        # Plots 6b, 6d, 6f
        #-----------------
        #
        dvariance = pvariance
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'b'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'd'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'f'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
    elif figure == 7:
        dmean = pmean
        dvar_range = 25
        dvariance = pvariance
        xtmarksMah = [4000, 6000, 8000, 10000, 12000, 14000]
        #
        #-----------------
        # Plots 7a, 7c, 7e
        #-----------------
        #
        dvar_type = 'sph'
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'a'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        DxMah = AlfOliModDiaX(prealiz)
        ecdfx = ECDF(DxMah)
        credLowX, credHighX = getCdfXLimits(pnumber, ecdfx, credLow, credHigh)
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'c'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DxMul = HaarModDia(prealiz, L) / x_size
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'e'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)
        #
        #-----------------
        # Plots 7b, 7d, 7f
        #-----------------
        #
        dvar_type = 'exp'
        #
        # Realizations
        #
        np.random.seed(seed=1)
        prealiz = generateRealizations(x_size, y_size, pvariance, pvar_range, aspect, angle, pvar_type, pmean, pnumber)
        drealiz = generateRealizations(x_size, y_size, dvariance, dvar_range, aspect, angle, dvar_type, dmean, dnumber)
        labelAllReas = str(figure) + 'b'
        plotAllReas(file, labelAllReas, dnumber, prealiz, drealiz, drealizplotno)
        #
        # Mahalanobis
        #
        ecdfz = getEcdfz(dnumber, prealiz, drealiz)
        labelMah = str(figure) + 'd'
        plotMah(file, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarksMah)
        #
        # Multiscale
        #
        DzMul = HaarModDia(drealiz, L) / x_size
        labelMul = str(figure) + 'f'
        pltType = 'meansAndStds'
        plotMul(file, labelMul, DxMul, DzMul, L, pltType)


def plotAllReas(f, label, pnumber, prealiz, drealiz, r):
    file = f + label + '.pdf'
    for i in np.arange(pnumber):
        plt.plot(prealiz[:, i])
    plt.plot(drealiz[:, r], 'k')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylim((-5, 5))
    xtmarks = [0, 200, 400, 600, 800, 1000]
    plt.xticks(xtmarks)
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def plotTwoReas(f, label, prealiz, drealiz, r):
    file = f + label + '.pdf'
    plt.plot(prealiz[:, r], 'r')
    plt.plot(drealiz[:, r], 'c')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylim((-5, 5))
    xtmarks = [0, 200, 400, 600, 800, 1000]
    plt.xticks(xtmarks)
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def plotMeansReas(f, labelMeansReas, pmean, dmean):
    file = f + labelMeansReas + '.pdf'
    plt.plot(pmean, 'r')
    plt.plot(dmean, 'c')
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.ylim((-5, 5))
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def plotMah(fileMah, labelMah, credLow, credHigh, ecdfx, ecdfz, credLowX, credHighX, xtmarks):
    file = fileMah + labelMah + '.pdf'
    plt.axhline(credLow, color='k')
    plt.axhline(credHigh, color='k')
    plt.plot(ecdfx.x, ecdfx.y, 'r .')
    plt.plot(ecdfz.x, ecdfz.y, 'c .')
    xzLow = np.array([ecdfx.x[1], ecdfz.x[1]])
    xzHigh = np.array([ecdfx.x[-1], ecdfx.x[-1]])
    pltlimLow = np.min(xzLow) - (ecdfx.x[5] - ecdfx.x[1])
    pltlimHigh = np.max(xzHigh) + (ecdfx.x[-1] - ecdfx.x[-5])
    xspan = [pltlimLow, pltlimHigh]
    yspan = [- 0.05, 1.05]
    ytmarks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    plt.fill_betweenx(yspan, xspan[0], credLowX, fc='pink')
    plt.fill_betweenx(yspan, credHighX, xspan[1], fc='pink')
    plt.xlim(xspan[0], xspan[1])
    plt.ylim(yspan[0], yspan[1])
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xticks(xtmarks)
    plt.yticks(ytmarks)
    plt.savefig(file, bbox_inches='tight')
    plt.close()


def plotMul(f, labelMul, DxMul, DzMul, L, pltType):
    alpha = 2
    file = f + labelMul + '.pdf'
    Dxmean = np.zeros(L)
    Dxstd = np.zeros(L)
    Dzmean = np.zeros(L)
    Dzstd = np.zeros(L)
    for l in np.arange(L):
        Dxmean[l] = np.mean(DxMul[:, l])
        Dxstd[l] = np.std(DxMul[:, l])
        Dzmean[l] = np.mean(DzMul[:, l])
        Dzstd[l] = np.std(DzMul[:, l])
    scale = 1
    DxmeanScaled = Dxmean / scale
    DxstdScaled = Dxstd / scale
    DzmeanScaled = Dzmean / scale
    DzstdScaled = Dzstd / scale
    DxmpsScaled = DxmeanScaled + alpha * DxstdScaled
    DxmmsScaled = DxmeanScaled - alpha * DxstdScaled
    DzmpsScaled = DzmeanScaled + alpha * DzstdScaled
    DzmmsScaled = DzmeanScaled - alpha * DzstdScaled
    xlcolor = 'silver'
    xlstyle = 'dotted'
    xlines = np.arange(L)
    xtmarks = xlines
    if pltType == 'stds':
        xlymin = 0
        xlymax = 1
        ylims = (xlymin, xlymax)
        ytmarks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        plt.plot(DxstdScaled, 'r')
        plt.plot(DzstdScaled, 'c')
        plt.vlines(xlines, xlymin, xlymax, colors=xlcolor, linestyles=xlstyle)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.ylim(ylims)
        plt.xticks(xtmarks)
        plt.yticks(ytmarks)
        plt.savefig(file, bbox_inches='tight')
        plt.close()
    else:
        xlymin = - 1.3
        xlymax = 1.3
        ylims = (xlymin, xlymax)
        ytmarks = [-1, -0.5, 0, 0.5, 1]
        plt.plot(DxmeanScaled, 'r')
        plt.plot(DzmeanScaled, 'c')
        plt.plot(DxmpsScaled, 'r--')
        plt.plot(DxmmsScaled, 'r--')
        plt.plot(DzmpsScaled, 'c--')
        plt.plot(DzmmsScaled, 'c--')
        plt.vlines(xlines, xlymin, xlymax, colors=xlcolor, linestyles=xlstyle)
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
        plt.ylim(ylims)
        plt.xticks(xtmarks)
        plt.yticks(ytmarks)
        plt.savefig(file, bbox_inches='tight')
        plt.close()


def generateRealizations(x_size, y_size, variance, var_range, aspect, angle, var_type, mean, number):
    #
    # Covariance matrix
    #
    cov = chol.genCov(x_size, y_size, variance, var_range, aspect, angle, var_type)
    #
    # Realizations
    #
    realiz = chol.gen_real(mean, cov, number)
    #
    return realiz


def getCdfXLimits(pnumber, ecdfx, credLow, credHigh):
    for i in np.arange(pnumber):
        if np.greater(ecdfx.y[i], credLow):
            iLowX = i
            break
    credLowX = ecdfx.x[iLowX]
    for i in np.arange(pnumber - 1, -1, -1):
        if np.less(ecdfx.y[i], credHigh):
            iHighX = i
            break
    credHighX = ecdfx.x[iHighX]
    #
    return credLowX, credHighX


def getEcdfz(dnumber, prealiz, drealiz):
    muz = np.zeros(dnumber)
    for i in np.arange(dnumber):
        muz[i] = AlfOliModDiaMultiZ(prealiz, drealiz[:, i])
    ecdfz = ECDF(muz)
    #
    return ecdfz


def AlfOliModDiaX(X):
    #
    ne = X.shape[1]
    ns = ne - 1
    I = np.eye(ns)
    nu = np.var(X)
    delta = 2 / (ns + 2)
    fac1 = delta * nu
    fac2 = fac1 * ns / (1 - delta)
    Mx = np.zeros(ns)
    for i in np.arange(ns):
        Xs = np.delete(X, i, 1)
        As = fac2 * I + np.matmul(np.transpose(Xs), Xs)
        Ls = np.linalg.cholesky(As)
        mu = np.sum(Xs, 1) / ns
        x  = X[:, i]
        dx = x - mu
        Mx[i] = approxMahalanobis(fac1, Xs, dx, Ls)
    #
    return Mx


def AlfOliModDiaMultiZ(X, z):
    #
    ne = X.shape[1]
    ns = ne - 1
    I = np.eye(ns)
    nu = np.var(X)
    delta = 2 / (ns + 2)
    fac1 = delta * nu
    fac2 = fac1 * ns / (1 - delta)
    Mz = np.zeros(ns)
    for i in np.arange(ns):
        Xs = np.delete(X, i, 1)
        As = fac2 * I + np.matmul(np.transpose(Xs), Xs)
        Ls = np.linalg.cholesky(As)
        mu = np.sum(Xs, 1) / ns
        dz = z - mu
        Mz[i] = approxMahalanobis(fac1, Xs, dz, Ls)
    muz = np.median(Mz)
    #
    return muz


def approxMahalanobis(fac, X, ds, L):
    #
    # Follows Alfonzo&Oliver, COMG 2019, pp 1331-1347
    #
    b = np.matmul(np.transpose(X), ds)
    y = np.linalg.solve(L, b)
    mdist = (np.matmul(np.transpose(ds), ds) - np.matmul(np.transpose(y), y)) / fac
    #
    return mdist


def HaarModDia(X, L):
    #
    ns = X.shape[1]
    Mx = np.zeros([ns, L])
    for i in np.arange(ns):
        x = X[:, i]
        Mx[i] = HaarProjections(x, L)
    #
    return Mx


def HaarProjections(x, L):
    #
    N = x.shape[0]
    M = np.zeros(L)
    for l in np.arange(L):
        h = HaarVector(N, l)
        M[l] = np.matmul(h, x)
    #
    return M


def HaarVector(N, l):
    #
    hVec = np.zeros(N).astype(int)
    expArg = l - 1
    if np.less(expArg, 0):
        hVec = np.ones(N).astype(int)
        return hVec
    K = 2 ** expArg
    konst = int(N / (2 * K))
    normalize = np.sqrt(K)
    lnullstop = - 2 * konst
    for k in np.arange(K):
        lnullstop = lnullstop + 2 * konst
        for r in np.arange(konst):
            hVec[lnullstop + r] = 1
            hVec[lnullstop + konst + r] = - 1
    hVec = normalize * hVec
    #
    return hVec


main()

