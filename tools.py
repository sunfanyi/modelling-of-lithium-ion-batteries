import numpy as np
import matplotlib.pyplot as plt

SOC_levels = np.arange(0.9, 0.1, -0.1)
SOC_from_idx_map = {i: SOC_levels[i] for i in range(8)}  # 8 SOC levels
temp_from_idx_map = {0: '$0^oC$', 1: '$20^oC$', 2: '$40^oC$'}  # 3 temperature level


def match_val(target, x, y):
    """
    Match the corresponding y value of a target value, where y = f(x).
    np.interp() is not used because it's computationally expensive and unstable
    for large arrays.

    eg., to find OCV(z=1), use match_val(1, z, OCV)
    """
    idx = np.abs(x - target).argmin()
    res = y[idx]

    return res


def select_pulse(t, I, V_actual, idx_pulse_start, idx_Vss,
                 SOC, pulse, pad_zero=True):
    """
    Extract one pulse from the whole graph
    """
    if pad_zero:
        # pad with some constant values both sides for visualisation
        idx_seg_start = idx_pulse_start[SOC-1, pulse-1] - 200
        idx_seg_end = idx_Vss[SOC-1, pulse-1] + 200
    else:
        idx_seg_start = idx_pulse_start[SOC-1, pulse-1]
        idx_seg_end = idx_Vss[SOC-1, pulse-1]

    t_seg = t[idx_seg_start : idx_seg_end+1]
    V_actual_seg = V_actual[idx_seg_start : idx_seg_end+1]
    I_seg = I[idx_seg_start : idx_seg_end+1]

    return t_seg, I_seg, V_actual_seg


def plot_pulses(t, I, V, idx_start, idx_end,
                SOC, pulse, idx_Vss=None, show_current=True, temperature=None):
    """
    For part 2 only.
    Used for plotting the training data only (8 SOC levels x 8 currents).
    Plot the voltage and current graph for selected SOC levels and current,with
     the current pulse starting and ending points and steady states labelled.
    Arrays can be for a single temperature (for part 2a) or three temperatures.
    :param t: time, [N] or [3 x N]
    :param I: current, [N] or [3 x N]
    :param V: voltage, [N] or [3 x N]
    :param idx_start: idx of current pulse starts, [8 x 8] or [3 x 8 x 8]
    :param idx_end: idx of current pulse starts, [8 x 8] or [3 x 8 x 8]
    :param SOC: integer 1 - 8 stating which SOC level to plot
    :param pulse: integer 1 - 8 stating which pulse (current) to plot, or 'all'
    :param idx_Vss: idx of steady state reached, [8 x 8] or [3 x 8 x 8]
    :param show_current: boolean, plot current is True
    :param temperature: which temp level, 0, 1, 2 -> 0, 20, 40 degC
    """
    if temperature is not None:
        t = t[temperature]
        temp_level = temp_from_idx_map[temperature]
        V = V[temperature]
        I = I[temperature]
        idx_start = idx_start[temperature]
        idx_end = idx_end[temperature]
        if idx_Vss is not None:
            idx_Vss = idx_Vss[temperature]
    else:
        temp_level = temp_from_idx_map[1]  # 20 deg

    SOC_level = SOC_from_idx_map[SOC - 1]
    if pulse == 'all':
        # for extrapolating:
        density = idx_end[SOC - 1, -1] - idx_start[SOC - 1, 0]
        xlim = [t[idx_start[SOC - 1, 0] - int(0.01*density)],
                t[idx_end[SOC - 1, -1] + int(0.15*density)]]
        ylim = [np.min(V[idx_start[SOC - 1, 0]:idx_end[SOC - 1, -1]]) - 0.05,
                np.max(V[idx_start[SOC - 1, 0]:idx_end[SOC - 1, -1]]) + 0.05]
        title = 'Temperature = {}\nSOC = {:0.0f}%'.format(
            temp_level, SOC_level*100)
    else:
        density = idx_end[SOC - 1, pulse - 1] - idx_start[SOC - 1, pulse - 1]
        xlim = [t[idx_start[SOC - 1, pulse - 1] - density],
                t[idx_end[SOC - 1, pulse - 1] + 8*density]]
        ylim = [np.min(V[idx_start[SOC - 1, pulse - 1]:
                         idx_end[SOC - 1, pulse - 1]]) - 0.05,
                np.max(V[idx_start[SOC - 1, pulse - 1]:
                         idx_end[SOC - 1, pulse - 1]]) + 0.05]
        title = 'Temperature = {}\nthe {}th pulse at SOC = {:0.0f}%'.format(
            temp_level, pulse, SOC_level * 100)

    if show_current:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax1 = axes[0]
        ax2 = axes[1]
    else:
        fig, ax1 = plt.subplots(figsize=(10, 4))

    # plot voltage
    ax1.plot(t, V)
    ax1.scatter(t[idx_start], V[idx_start],
                c='r', s=20, marker='x', label='Pulse start/end')
    ax1.scatter(t[idx_end], V[idx_end],
                c='r', s=20, marker='x')
    if idx_Vss is not None:
        ax1.scatter(t[idx_Vss], V[idx_Vss],
                    c='g', s=20, marker='x', label='Steady State')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')

    # plot current
    if show_current:
        ax2.plot(t, I)
        ax2.scatter(t[idx_start], I[idx_start],
                    c='r', s=20, marker='x', label='Pulse start/end')
        ax2.scatter(t[idx_end], I[idx_end],
                    c='r', s=20, marker='x')
        if idx_Vss is not None:
            ax2.scatter(t[idx_Vss], I[idx_Vss],
                        c='g', s=20, marker='x', label='Steady State')
        ax2.set_xlim(xlim)
        # ax2.set_ylim(ylim)
        ax2.legend()
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Current (A)')

    fig.suptitle(title)
    plt.show()


def find_Vss_pos(t, V, idx_pulse_start, idx_pulse_end,
                 threshold=500, for_bump=False):
    """
    For part 2 only.
    Arrays can be for a single temperature (for part 2a) or three temperatures.
    Find the index where the steady state is reached for each pulse.
    :param t: time, [N] or [3 x N]
    :param V: voltage, [N] or [3 x N]
    :param idx_pulse_start: idx of current pulse starts, [8 x 8] or [3 x 8 x 8]
    :param idx_pulse_end: idx of current pulse starts, [8 x 8] or [3 x 8 x8]
    :param threshold: criteria for judging the steady state: the steady state is
        reached if 'threshold' of constant values are observed
    :param for_bump: if true: used for the small bump before each 8 pulses only
    :return:
    """
    if len(t.shape) == 1:  # expand one dimension representing temperature
        t = np.expand_dims(t, 0)
        V = np.expand_dims(V, 0)
        idx_pulse_start = np.expand_dims(idx_pulse_start, 0)
        idx_pulse_end = np.expand_dims(idx_pulse_end, 0)

    num_temp = V.shape[0]
    idx_Vss = []
    for j in range(num_temp):
        # for each temperature value:
        idx_each_temp = []
        all_ends = idx_pulse_end[j].flatten()
        for pos in range(len(all_ends)):
            i = 0
            while True:
                segment = V[j, all_ends[pos] + i: all_ends[pos] + i + threshold]
                if np.all(segment == segment[0]):
                    # all element in this array are the same
                    idx_each_temp.append(all_ends[pos] + i)
                    break
                i += 1

                # Stop browsing if touching the next next pulse, or the end
                i_limit = np.concatenate(
                    (idx_pulse_start[j].flatten()[1:], [t.shape[1]]))
                if i > i_limit[pos]:
                    which_SOC = pos // 8
                    SOC_level = SOC_from_idx_map[which_SOC]
                    msg = "Position of V_ss is not found for the {}th pulse at SOC = {:0.0f}%".format(
                        pos % 8, SOC_level * 100)
                    raise ValueError(msg)

        if for_bump:  # return 8 values for each temperature
            idx_Vss.append(idx_each_temp)
        else:  # return 64 values for each temperature
            idx_Vss.append(np.reshape(idx_each_temp, [8, 8]))

    return np.array(idx_Vss)


def para_RC(t, I, V, idx_pulse_end, idx_Vss):
    """
    Used for part 2.
    Parametrisation for R0, R1 and C1, as training data.
    """
    if len(t.shape) == 1:  # only for one temperature
        V_peaks = V[idx_pulse_end - 1]
        I_peaks = I[idx_pulse_end - 1]

        d_V0 = np.abs(V[idx_pulse_end] - V_peaks)
        d_I = np.abs(I[idx_pulse_end] - I_peaks)
        R0_tab = d_V0 / d_I

        d_Vinf = np.abs(V[idx_Vss] - V_peaks)
        R1_tab = d_Vinf / d_I - R0_tab

        C1_tab = np.abs((t[idx_Vss] - t[idx_pulse_end]) / (4 * R1_tab))

    else:  # for multiple (e.g., 3) temperatures
        def index_3D(array, idx):
            res = [np.reshape(array[i][idx.reshape(3, -1)[i]], [8, 8]) for i in
                   range(3)]
            return np.array(res)

        V_peaks = index_3D(V, idx_pulse_end - 1)
        I_peaks = index_3D(I, idx_pulse_end - 1)

        V_pulse_end = index_3D(V, idx_pulse_end)
        d_V0 = np.abs(V_pulse_end - V_peaks)
        I_pulse_end = index_3D(I, idx_pulse_end)
        d_I = np.abs(I_pulse_end - I_peaks)
        R0_tab = d_V0 / d_I

        V_Vss = index_3D(V, idx_Vss)
        d_Vinf = np.abs(V_Vss - V_peaks)
        R1_tab = d_Vinf / d_I - R0_tab

        t_Vss = index_3D(t, idx_Vss)
        t_pulse_end = index_3D(t, idx_pulse_end)
        C1_tab = np.abs((t_Vss - t_pulse_end) / (4 * R1_tab))

    return R0_tab, R1_tab, C1_tab, I_peaks, d_I, d_Vinf


def update_SOC(i, z, t, I, eta, Q):
    """
    Run this function in a loop, update the state of charge
    """
    # *100 to convert to %
    z[i+1] = z[i] - 100 * I[i]*eta*(t[i+1]-t[i]) / (Q/1000*3600)


def update_I_R1(i, I_R1, t, I, R1, C1):
    """
    For the first order ECN Model in Part 2
    Run this function in a loop, update the I_R1
    """
    dt = t[i+1] - t[i]
    a = -1 / (R1*C1)
    b = 1 / (R1*C1)
    I_R1[i+1] = np.exp(a*dt) * I_R1[i] + \
                1/a * (np.exp(a*dt)-1) * b * I[i]


def first_order_ECN(t, I, T, V_actual, ref_OCV, ref_SOC,
                    fit_R0, fit_R1, fit_C1):
    """
    The first order ECN model for part 2. t is time and T is temperature.
    !!!!!
    fit_R0, fit_R1, fit_C1 are placeholder functions for fitting R0, R1, C1.
    They need to be defined before calling first_order_ECN().
    They need to be changed as the model updating.
    They are created to ensure first_order_ECN() can be used
        everywhere throughout this project.
    """
    Q = 2500
    eta = 1

    N = len(t)
    z = np.ndarray([N, 1])
    V_pred = np.ndarray([N, 1])
    OCV = np.ndarray([N, 1])
    I_R1 = np.ndarray([N, 1])
    if T is None:
        # redundant array, to prevent error when calling T[i]
        T = np.ndarray([N, 1])

    z0 = match_val(V_actual[0], ref_OCV, ref_SOC)
    z[0] = z0
    I_R1[0] = 0

    for i in range(N):
        R1_val = fit_R1(I[i], z[i], T[i])  # use values at i
        R0_val = fit_R0(I[i], z[i], T[i])

        OCV[i] = match_val(z[i], ref_SOC, ref_OCV)
        V_pred[i] = OCV[i] - R1_val * I_R1[i] - R0_val * I[i]

        if i != N-1:
            update_SOC(i, z, t, I, eta, Q)  # update z at i+1
            # We are updating the next value (i+1):
            R1_val = fit_R1(I[i+1], z[i+1], T[i+1])  # use values at i+1
            C1_val = fit_C1(I[i+1], z[i+1], T[i+1])

            update_I_R1(i, I_R1, t, I, R1_val, C1_val)  # update I_R1 at i+1
    return V_pred


def first_order_ECN_temp(t, I, T_init, V_actual, ref_OCV, ref_SOC,
                    fit_R0_temp, fit_R1_temp, fit_C1_temp, T_change):
    """
    The first order ECN model for part 2. t is time and T is temperature.
    !!!!!
    fit_R0, fit_R1, fit_C1 are placeholder functions for fitting R0, R1, C1.
    They need to be defined before calling first_order_ECN().
    They need to be changed as the model updating.
    They are created to ensure first_order_ECN() can be used
        everywhere throughout this project.
    """
    Q = 2500
    eta = 1

    N = len(t)
    z = np.ndarray([N, 1])
    V_pred = np.ndarray([N, 1])
    OCV = np.ndarray([N, 1])
    I_R1 = np.ndarray([N, 1])

    T = np.ndarray([N, 1])


    z0 = match_val(V_actual[0], ref_OCV, ref_SOC)
    z[0] = z0
    I_R1[0] = 0

    T_new = T_init      # Initial Cell Temperature

    for i in range(N):
        dt = t[i+1] - t[i]          # Time step
        # dt = 1
        R0_val = fit_R0_temp(T_new)  # use values at i
        R1_val = fit_R1_temp(I[i], T_new)
        T_new = T_change(I[i], R0_val, R1_val, dt, T_new)

        OCV[i] = match_val(z[i], ref_SOC, ref_OCV)
        V_pred[i] = OCV[i] - R1_val * I_R1[i] - R0_val * I[i]

        if i != N-1:
            update_SOC(i, z, t, I, eta, Q)  # update z at i+1
            # We are updating the next value (i+1):
            R0_val = fit_R0_temp(T[i+1])
            R1_val = fit_R1_temp(I[i], T[i+1])  # use values at i+1
            C1_val = fit_C1_temp(T[i+1])

            update_I_R1(i, I_R1, t, I, R1_val, C1_val)  # update I_R1 at i+1
    return V_pred
