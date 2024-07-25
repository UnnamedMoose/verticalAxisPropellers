import pandas
import numpy as np
import os
import json
import copy
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import matplotlib

font = {"family": "serif",
        "weight": "normal",
        "size": 16}

matplotlib.rc("font", **font)
matplotlib.rcParams["figure.figsize"] = (9, 6)

def makeNiceAxes(ax, xlab=None, ylab=None, invisibley="right"):
    ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)
    ax.spines[invisibley].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

# %% Routines for computing the kinematics of a vertical-axis propeller.
def motionParams(D, cByD, rpm, Va, N, tByT, refBladePos=0., xcRot=0.25, rho=1025.):
    omega = rpm/60 * (2 * np.pi)

    # Position of the centre of the blade. Zero degrees at x=0 and negative y, consistently
    # with the convention shown in the Voith-Schneider documentation. Rotation +ve anticlockwise.
    psi = 2.*np.pi*tByT + refBladePos
    x = D/2. * np.sin(psi)
    y = -D/2. * np.cos(psi)

    # In vector form.
    P = np.array((x, y))
    O = np.array([0., 0.])

    # Vectors from the centre of effort and centre of rotation to the blade position.
    v_np = np.array([P[0] - N[0], P[1] - N[1]])
    v_op = np.array([P[0] - O[0], P[1] - O[1]])
    # Vector defining the chordline (+ve in the TE direction, i.e. along x/c).
    v_tan = np.array([v_np[1], -v_np[0]]) / np.linalg.norm(v_np)
    # Flow direction due to the motion of the ship.
    v_ship = np.array([Va, 0.])
    # Local velocity due to rotation.
    v_rot = -omega*D/2.*np.array([v_op[1], -v_op[0]])/np.linalg.norm(v_op)
    # Total velocity vector.
    v_flow = v_ship + v_rot

    # leading edge position.
    xy_LE = P - xcRot*v_tan*cByD*D

    # Angle of attack, ignoring induced velocity.
    cos_alpha = np.clip(np.dot(v_flow, -v_tan) / np.linalg.norm(v_flow), -1., 1.)
    alpha = np.arccos(cos_alpha)
    # Calculate the cross product to determine the direction
    cross_product = np.cross(v_flow, -v_tan)
    # Determine the sign of the angle based on the cross product's z-component
    if cross_product < 0:
        alpha = -alpha

    # Rotation angle of the blade around its pivot point from the reference position.
    # The same as the angle of attack when ignoring ship speed.
    alpha_mod = np.degrees(np.arctan2(v_np[1], v_np[0]) - np.arctan2(v_op[1], v_op[0]))
    if alpha_mod < -180:
        alpha_mod += 360
    elif alpha_mod > 180:
        alpha_mod -= 360
    alpha_geom = np.radians(alpha_mod)

    # Rotation angle of the blade in the global reference frame. +pi when pointing along +ve x
    # and wraps around at -pi.
    psi = np.arctan2(-v_tan[1], v_tan[0])

    # TODO
    Cl = 2.*np.pi * alpha
    Cd = 0.01

    Drag = -Cd*0.5*rho*(cByD*D)**2.*np.linalg.norm(v_flow)**2. * v_flow/np.linalg.norm(v_flow)
    Lift = Cl*0.5*rho*(cByD*D)**2.*np.linalg.norm(v_flow)**2. * np.array([-v_flow[1], v_flow[0]])//np.linalg.norm(v_flow)

    return P, xy_LE, v_tan, v_ship, v_rot, v_flow, psi, alpha_geom, alpha, Lift, Drag

def evalTimeSeries(tVals, D, cByD, rpm, Va, N):
    data = []
    for t in tVals:
        xy_rot, xy_LE, v_tan, v_ship, v_rot, v_flow, psi, alpha_geom, alpha, Lift, Drag = motionParams(D, cByD, rpm, Va, N, t)
        data.append(dict(zip(
            ["t", "x", "y", "x_LE", "y_LE", "vx_tan", "vy_tan", "vx_ship", "vy_ship",
             "vx_rot", "vy_rot", "vx_flow", "vy_flow", "psi", "alpha_geom", "alpha", "Lx", "Ly", "Dx", "Dy"],
            [t, xy_rot[0], xy_rot[1], xy_LE[0], xy_LE[1], v_tan[0], v_tan[1], v_ship[0], v_ship[1],
             v_rot[0], v_rot[1], v_flow[0], v_flow[1], psi, alpha_geom, alpha, Lift[0], Lift[1], Drag[0], Drag[1]]
        )))
    return pandas.DataFrame(data)

# %% Make an interactive tool for visualising the regressed wake and the nearest training data.
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(224)
ax4 = ax3.twinx()
plt.subplots_adjust(top=0.925, bottom=0.105, left=0.045, right=0.905, hspace=0.44, wspace=0.21)

makeNiceAxes(ax1, "x [m]", "y [m]")
makeNiceAxes(ax2, "x$_{Flow \; fixed}$ [m]", "y [m]")
makeNiceAxes(ax3, "t/T", "$\\alpha$ [deg]")
makeNiceAxes(ax4, "t/T", "$\\psi$ [deg]", invisibley="left")

cs = None
text = None

# Set the interactive things.
sliders = []
sliderLocs = [
    [0.1, 0.33, 0.25, 0.05],
    [0.1, 0.27, 0.25, 0.05],
]
for i, x in enumerate(["x", "y"]):
    sldr_ax = fig.add_axes(sliderLocs[i])
    sliders.append(Slider(sldr_ax, 'N{}/D'.format(x), -0.6, 0.6, valinit=0., valfmt="%.1f"))

boxLocs = [
    [0.1, 0.03, 0.15, 0.04],
    [0.1, 0.09, 0.15, 0.04],
    [0.1, 0.15, 0.15, 0.04],
    [0.1, 0.21, 0.15, 0.04],
]
textBoxes = []

textBoxes.append(TextBox(fig.add_axes(boxLocs[0]), "D [m]", textalignment="center"))
textBoxes[-1].set_val("2.6")

textBoxes.append(TextBox(fig.add_axes(boxLocs[1]), "c/D", textalignment="center"))
textBoxes[-1].set_val("0.2")

textBoxes.append(TextBox(fig.add_axes(boxLocs[2]), "rpm", textalignment="center"))
textBoxes[-1].set_val("30")

textBoxes.append(TextBox(fig.add_axes(boxLocs[3]), "Va [m/s]", textalignment="center"))
textBoxes[-1].set_val("1.0")

values = {
    "D": 2.6,
    "cByD": 0.2,
    "rpm": 30,
    "Va": 1.0,
    "NxByD": 0.0,
    "NyByD": 0.0,
}

# Update routine
def update():
    global cs
    global values
    global text
    if cs is not None:
        try:
            for c in cs.collections:
                c.remove()
        except AttributeError:
            for c in cs:
                c.remove()
    cs = []

    # Get user controls.
    D = values["D"]
    cByD = values["cByD"]
    rpm = values["rpm"]
    Va = values["Va"]
    N = np.array([values["NxByD"], values["NyByD"]])*D

    advCoeff = Va / (np.pi*rpm/60*D)
    eccentricity = np.linalg.norm(N) / (D/2.)

    # TODO add text somewhere
    if text is None:
        text = ax3.text(-1.07, 0.85, "e={:.3f} $\lambda$={:.3f}".format(eccentricity, advCoeff),
            ha="left", transform=ax3.transAxes)
    else:
        text.set_text("e={:.3f} $\lambda$={:.3f}".format(eccentricity, advCoeff))

    # Evaluate the kinematics.
    data = evalTimeSeries(np.linspace(0, 1, 11), D, cByD, rpm, Va, N)
    dataFine = evalTimeSeries(np.linspace(0, 1, 201), D, cByD, rpm, Va, N)

    # Plot the motion in a ship-fixed reference frame.
    ax = ax1
    ax.set_aspect("equal")
    cs += ax.plot(dataFine["x"], dataFine["y"], "k-", lw=2, alpha=0.5)
    cs += ax.plot(0, 0, "ko")
    cs += ax.plot(N[0], N[1], "mp", label="N", ms=9, zorder=1000)

    for i in range(data.shape[0]):
        lns = ax.plot(np.array([0., data.loc[i, "vx_tan"]*D*cByD])+data.loc[i, "x_LE"],
                      np.array([0., data.loc[i, "vy_tan"]*D*cByD])+data.loc[i, "y_LE"],
                      "k-", lw=2, label="Chordline")

        arrowScale = 0.2
        vscale = np.linalg.norm(data.loc[i, ["vx_ship", "vy_ship"]])

        arrowScale = 0.3 / np.max(np.linalg.norm(data[["Lx", "Ly"]].values, axis=1))
        vec = data.loc[i, ["Lx", "Ly"]].values*arrowScale
        lns.append(ax.arrow(data.loc[i, "x"], data.loc[i, "y"], vec[0], vec[1], width=0.02, fc="r", ec="r", label="Lift"))

        arrowScale = 0.3 / np.max(np.linalg.norm(data[["Dx", "Dy"]].values, axis=1))
        vec = data.loc[i, ["Dx", "Dy"]].values*arrowScale
        lns.append(ax.arrow(data.loc[i, "x"], data.loc[i, "y"], vec[0], vec[1], width=0.02, fc="b", ec="b", label="Drag"))

        cs += lns
        cs += ax.plot([N[0], data.loc[i, "x"]], [N[1], data.loc[i, "y"]], "k-", lw=1, alpha=0.25)
        cs += ax.plot(data.loc[i, "x"], data.loc[i, "y"], "k.", ms=9)

    ax.legend(lns, [l.get_label() for l in lns], loc="lower left", bbox_to_anchor=(0., 1.01), ncol=3)

    # Plot the motion in the fluid-fixed reference frame.
    ax = ax2
    ax.set_aspect("equal")
    cs += ax.plot(dataFine["t"]*Va*(1./(rpm/60.))+dataFine["x"], dataFine["y"], "k-", lw=2, alpha=0.5)
    for i in range(data.shape[0]):
        x = data.loc[i, "t"]*Va*(1./(rpm/60.))+data.loc[i, "x_LE"]
        lns = ax.plot(np.array([0., data.loc[i, "vx_tan"]*D*cByD])+x,
                      np.array([0., data.loc[i, "vy_tan"]*D*cByD])+data.loc[i, "y_LE"],
                      "k-", lw=2, label="Chordline")

        x = data.loc[i, "t"]*Va*(1./(rpm/60.))+data.loc[i, "x"]
        arrowScale = 0.3 / np.max(np.linalg.norm(data[["Lx", "Ly"]].values, axis=1))
        vec = data.loc[i, ["Lx", "Ly"]].values*arrowScale
        lns.append(ax.arrow(x, data.loc[i, "y"], vec[0], vec[1], width=0.02, fc="r", ec="r", label="Lift"))

        arrowScale = 0.3 / np.max(np.linalg.norm(data[["Dx", "Dy"]].values, axis=1))
        vec = data.loc[i, ["Dx", "Dy"]].values*arrowScale
        lns.append(ax.arrow(x, data.loc[i, "y"], vec[0], vec[1], width=0.02, fc="b", ec="b", label="Drag"))

        cs += lns

    # Plot the variation of the angle of attack.
    lns = ax3.plot(dataFine["t"], dataFine["alpha_geom"]/np.pi*180, "b-", lw=2, label="Geometric AoA")
    lns += ax3.plot(dataFine["t"], dataFine["alpha"]/np.pi*180, "r-", lw=2, label="True AoA")
    lns += ax4.plot(dataFine["t"], dataFine["psi"]/np.pi*180, "g-", lw=2, label="Pitch angle")
    ax3.legend(lns, [l.get_label() for l in lns], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    cs += lns

    return cs

# Bind to the data parsing and plot update routines
def set_N(val):
    global values; values["NxByD"] = sliders[0].val; values["NyByD"] = sliders[1].val; update()
def set_D(txt):
    global values; values["D"] = float(txt); update()
def set_cByD(txt):
    global values; values["cByD"] = float(txt); update()
def set_rpm(txt):
    global values; values["rpm"] = float(txt); update()
def set_Va(txt):
    global values; values["Va"] = float(txt); update()

sliders[0].on_changed(set_N)
sliders[1].on_changed(set_N)
textBoxes[0].on_submit(set_D)
textBoxes[1].on_submit(set_cByD)
textBoxes[2].on_submit(set_rpm)
textBoxes[3].on_submit(set_Va)

# Initialise with default values.
cs = update()

plt.show()
