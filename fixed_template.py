import logging
import matplotlib.pyplot as plt
import numpy as np
 
log = logging.getLogger(__name__)
 
# font to be used for titles, axes labels, and tick-label sizes
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}
 
#from profilehooks import profile
#@profile
def plotAlignmentDensityByTemplatePosition(cmph5, mh_name):
    tEnds = cmph5.tEnd
    tStarts = cmph5.tStart
    density_ends, bin_edges = np.histogram(tEnds, density=True,
                                           bins=range(0, 25000, 500))
    density_starts, bin_edges = np.histogram(tStarts, density=True,
                                             bins=range(0, 25000, 500))
    centers = []
    for index, val in enumerate(bin_edges):
        if index < len(bin_edges)-1:
            centers.append(float(bin_edges[index] + bin_edges[index+1])/2)
 
    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(np.array(centers)/1000., density_starts, lw=3)
    ax.plot(np.array(centers)/1000., density_ends, lw=3)
    ax.set_xlabel('Template position (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('Density', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.legend(('tStart', 'tEnd'), prop={'size':16})
    ax.set_title(mh_name, fontdict=font)
    return fig, ax
 
def plot_tEnd_vs_tStart(cmph5, mh_name):
    tEnds = cmph5.tEnd / 1000.
    tStarts = cmph5.tStart / 1000.
 
    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(tStarts, tEnds, alpha=0.25, marker='.', color='blue', s=30)
    ax.set_xlabel('tStart (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('tEnd (kb)', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.set_title(mh_name, fontdict=font)
    return fig, ax
 
def plot_templatePositionSurvival(cmph5, mh_name):
    # survival curve
    templatePositions = range(min(cmph5.tStart), max(cmph5.tEnd), 1)
    template_survival_dictionary = {}
    for pos in templatePositions:
        template_survival_dictionary[pos] = 0
 
    for alignment in cmph5:
        positions = np.unique(alignment.referencePositions())
        for position in positions:
            template_survival_dictionary[position] += 1
    x = np.array(template_survival_dictionary.keys())
    y = []
    for position in x:
        y.append(template_survival_dictionary[position])
    y = np.array(y)
    cg_x = [] # coarse-grain
    cg_y = []
    cg_window = 100
    x_sum = 0 # sums for coarse-graining
    y_sum = 0
    for index, val in enumerate(x):
        if index % cg_window == 0:
            cg_x.append(float(x_sum)/float(cg_window))
            cg_y.append(float(y_sum)/float(cg_window))
            x_sum = 0
            y_sum = 0
        else:
            x_sum += val
            y_sum += y[index]
 
    fig = plt.figure(figsize=(20, 6), dpi=200)
    ax = fig.add_subplot(121)
    ax.plot(np.array(cg_x)/1000., cg_y, color='blue', lw=3)
    ax.set_xlabel('Template Position (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('# of Survivors', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.set_title(mh_name, fontdict=font)
 
    # termination rate
    dx = np.diff(cg_x)
    dy = np.diff(cg_y)
    np.seterr(divide='ignore', invalid='ignore')
    dydx = np.divide(dy, dx, dtype=float)
    dydx_over_S = np.divide(dydx, cg_y[1:], dtype=float)
 
    ax = fig.add_subplot(122)
    ax.plot(np.array(cg_x[1:])/1000., -dydx_over_S*100., lw=3)
    ax.set_xlabel('Template Position (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('Termination rate (% per bp)', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.set_title(mh_name, fontdict=font)
    ymax_index = cg_y.index(np.max(cg_y))
    xlim_max = np.array(cg_x)[1:][np.max(
        [index for index, val in enumerate(cg_y[1:])
         if val > 0.1*np.max(cg_y[1:]) and index > ymax_index+1])]/1000.
    ax.set_xlim([0, xlim_max])
    rate = np.array(-dydx_over_S*100.)[np.where(
        np.array(cg_x)/1000. < xlim_max)]
    ylim_max = np.max(rate)
    ax.set_ylim([0, ylim_max*1.1])
    return (fig, ax)
 
def _getZMWRead(basecalls, zmw):
    # find which bax file the zmw is in
    for reader in basecalls.parts:
        if (np.min(reader.allSequencingZmws) <= zmw and
                np.max(reader.allSequencingZmws) >= zmw):
            baxreader = reader
            break
    return basecalls.ZmwRead(baxreader, zmw, 0, basecalls[zmw].numEvents)
 
def _getFrameRate(basecalls):
    bax_part = basecalls.parts[0]
    h5peer = bax_part.file
    scan_data = h5peer.get('ScanData')
    acq_params_attrs = scan_data.get('AcqParams').attrs
    frame_rate = acq_params_attrs['FrameRate']
    if type(frame_rate) is np.ndarray: # legacy
        frame_rate = frame_rate[0]
    elif type(frame_rate) is np.float32: # current
        pass
    return frame_rate
 
def plot_tStart_vs_alnTimeStart(cmph5, bash5, plsh5, mh_name):
    alignment_tStart = []
    alignment_start_time = []
    for aln in cmph5:
        zmw = aln.HoleNumber
        rstart = aln.rStart
        zmw_read = _getZMWRead(bash5, zmw)
        pulse_ix = zmw_read.PulseIndex()[rstart]
        zmw_pulses = plsh5[zmw].pulses()
        alignment_tStart.append(aln.tStart)
        alignment_start_time.append(zmw_pulses.pulseStartFrame()[pulse_ix])
 
    frame_rate = _getFrameRate(bash5)
 
    fig = plt.figure(figsize=(14, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.scatter(np.array(alignment_start_time)/(frame_rate*60.),
               np.array(alignment_tStart)/1000.,
               alpha=0.25, marker='.', color='blue', s=30)
    ax.set_xlabel('Alignment start time (minutes)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('tStart (kb)', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.set_title(mh_name, fontdict=font)
    return fig, ax
 
def plot_tSpan_vs_tStart_or_tEnd(cmph5, mh_name):
    tEnds = cmph5.tEnd
    tStarts = cmph5.tStart
    tSpans = tEnds - tStarts
 
    fig = plt.figure(figsize=(14, 16), dpi=200)
    ax = fig.add_subplot(211)
    ax.set_title(mh_name, fontdict=font)
    ax.scatter(tStarts/1000., tSpans/1000.,
               alpha=0.25, marker='.', color='red', s=15)
    ax.set_xlim([-0.5, np.max(tEnds)/1000.])
    ax.set_ylim([-0.5, np.max(tSpans)/1000.])
    ax.set_xlabel('tStart (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('Template span (kb)', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    ax = fig.add_subplot(212)
    ax.scatter(tEnds/1000., tSpans/1000.,
               alpha=0.25, marker='.', color='blue', s=15)
    ax.set_xlim([-0.5, np.max(tEnds)/1000.])
    ax.set_ylim([-0.5, np.max(tSpans)/1000.])
    ax.set_xlabel('tEnd (kb)', fontdict=font)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.set_ylabel('Template span (kb)', fontdict=font)
    ax.yaxis.set_tick_params(labelsize=22)
    return fig, ax