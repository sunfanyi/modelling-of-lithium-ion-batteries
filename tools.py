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
        xlim = [t[idx_start[SOC - 1, 0] - 1000],
                t[idx_end[SOC - 1, -1] + 4000]]
        ylim = [np.min(V[idx_start[SOC - 1, 0]:idx_end[SOC - 1, -1]]) - 0.05,
                np.max(V[idx_start[SOC - 1, 0]:idx_end[SOC - 1, -1]]) + 0.05]
        title = 'Temperature = {}\nSOC = {:0.0f}%'.format(
            temp_level, SOC_level*100)
    else:
        xlim = [t[idx_start[SOC - 1, pulse - 1] - 200],
                t[idx_end[SOC - 1, pulse - 1] + 5000]]
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

